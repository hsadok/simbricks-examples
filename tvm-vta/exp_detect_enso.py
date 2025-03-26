import os
import sys
import itertools
import tarfile

import typing as tp

sys.path.append(str(os.path.dirname(__file__)))

from simbricks.orchestration.experiments import Experiment
from simbricks.orchestration.simulators import (
    EnsoBMNIC,
    HostSim,
    NetSim,
    PCISwitchSim,
    QemuHost,
    QemuIcountHost,
    SwitchNet,
)
from simbricks.orchestration.nodeconfig import AppConfig, NodeConfig, EnsoNode

import exp_util


class VtaEnsoNode(exp_util.VtaNode):
    def __init__(self) -> None:
        super().__init__()
        self.memory = 16 * 1024

        self.enso_parent_dir = "/root"
        self.local_enso_dir: tp.Optional[str] = None
        self.enso_vm_mount_name = "enso"
        self.app_vm_mount_name = "app"

    @property
    def enso_dir(self) -> str:
        return f"{self.enso_parent_dir}/{self.enso_vm_mount_name}"
    
    @property
    def app_dir(self) -> str:
        return f"{self.enso_parent_dir}/{self.app_vm_mount_name}"

    def prepare_pre_cp(self) -> list[str]:
        cmds = super().prepare_pre_cp()
        cmds.extend(
            [
                "mount -t proc proc /proc",
                "mount -t sysfs sysfs /sys",
            ]
        )
        if self.local_enso_dir is not None:
            cmds.extend(
                [
                    f"mkdir -p {self.enso_parent_dir}",
                    (
                        f"tar xf /tmp/guest/{self.enso_vm_mount_name}"
                        f" -C {self.enso_parent_dir}"
                    ),
                ]
            )

        # Extract application.
        cmds.extend(
            [
                f"mkdir -p {self.enso_parent_dir}",
                (
                    f"tar xf /tmp/guest/{self.app_vm_mount_name}"
                    f" -C {self.enso_parent_dir}"
                ),
            ]
        )

        cmds.extend(
            [
                f"cd {self.enso_dir}",
                "./scripts/sw_setup.sh 16384 32768 true",
                "lsmod",
            ]
        )

        # Compile app.
        cmds.extend(
            [
                f"cd {self.app_dir}",
                "mkdir -p build",
                "cd build",
                "cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..",
                "make",
            ]
        )
        return cmds

    def prepare_post_cp(self) -> list[str]:
        cmds = super().prepare_post_cp()
        cmds.extend(
            [
                f"cd {self.enso_dir}/software/kernel/linux/",
                "export SHELLOPTS",
                "bash -x ./install",
            ]
        )

        return cmds

    # pylint: disable=consider-using-with
    def config_files(self, environment):
        files = super().config_files(environment)
        tar_path = f"/tmp/{self.app_vm_mount_name}.tar"
        
        local_app_dir = os.path.join(os.path.dirname(__file__), "enso_vta")
        with tarfile.open(tar_path, "w") as tar:
            tar.add(local_app_dir, arcname=self.app_vm_mount_name)

        files[self.app_vm_mount_name] = open(tar_path, "rb")

        if self.local_enso_dir is not None:
            # Tar the directory to be copied to the VM.
            tar_path = f"/tmp/{self.enso_vm_mount_name}.tar"
            with tarfile.open(tar_path, "w") as tar:
                tar.add(self.local_enso_dir, arcname=self.enso_vm_mount_name)

            files[self.enso_vm_mount_name] = open(tar_path, "rb")

        return files


class TvmVtaCpuEnsoServer(AppConfig):
    """Receive images through Enso and send them to VTA for inference."""

    def __init__(self, pci_device_id):
        super().__init__()
        self.pci_device_id = pci_device_id
        self.debug = False
        """Whether to dump inference result."""

    def run_cmds(self, node):
        cmds = [
            "cd /root/tvm/build/",
            (
                f"VTA_DEVICE={self.pci_device_id} "
                "LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH "
                "gdb --batch -x /root/tvm/cmds.gdb --args "
                "/root/app/build/enso_vta_server --model=/root/darknet "
                "--device=extdev --dump-meta"
            ),
        ]

        # dump image with detection boxes as base64 to allow later inspection
        if self.debug:
            cmds.extend(
                [
                    "echo dump deploy_detection-infer-result.png START",
                    "base64 deploy_detection-infer-result.png",
                    "echo dump deploy_detection-infer-result.png END",
                ]
            )
        return cmds
    

class TvmVtaCpuEnsoClient(AppConfig):
    """Receive images through Enso and send them to VTA for inference."""

    def __init__(self):
        super().__init__()
        self.debug = False
        """Whether to dump inference result."""

    def run_cmds(self, node):
        cmds = [
            "sleep 5",
            "cd /root/tvm/build/",
            (
                "LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH "
                "gdb --batch -x /root/tvm/cmds.gdb --args "
                "/root/app/build/enso_vta_client --model=/root/darknet "
                "--device=extdev --dump-meta"
            ),
        ]

        # dump image with detection boxes as base64 to allow later inspection
        if self.debug:
            cmds.extend(
                [
                    "echo dump deploy_detection-infer-result.png START",
                    "base64 deploy_detection-infer-result.png",
                    "echo dump deploy_detection-infer-result.png END",
                ]
            )
        return cmds


def add_server(
    experiment: Experiment,
    HostClass: tp.Type[QemuHost],
    app_config: AppConfig,
    network: NetSim,
    sync: bool,
    vta_clk_freq: int,
) -> HostSim:
    server_cfg = VtaEnsoNode()
    server_cfg.local_enso_dir = "/enso"
    server_cfg.app = app_config
    server = HostClass(server_cfg)
    server.sync = sync
    server.name = "s"

    nic = EnsoBMNIC()
    nic.set_network(network)
    nic.name = "nic"

    pci_sw = PCISwitchSim()
    pci_sw.name = "psw"

    vta = exp_util.VTADev()
    vta.clock_freq = vta_clk_freq
    vta.name = "vta"

    server.add_pcidev(pci_sw)
    pci_sw.add_pcidev(vta)
    pci_sw.add_nic(nic)

    experiment.add_nic(nic)
    experiment.add_pcidev(vta)
    experiment.add_pcidev(pci_sw)
    experiment.add_host(server)

    return server


def add_client(
    experiment: Experiment,
    HostClass: tp.Type[QemuHost],
    app_config: AppConfig,
    network: NetSim,
    sync: bool,
) -> HostSim:
    client_cfg = VtaEnsoNode()
    client_cfg.local_enso_dir = "/enso"
    client_cfg.app = app_config
    client = HostClass(client_cfg)
    client.sync = sync
    client.name = "c"

    nic = EnsoBMNIC()
    nic.set_network(network)
    nic.name = "nic"

    client.add_nic(nic)

    experiment.add_nic(nic)
    experiment.add_host(client)

    return client


experiments = []

# Experiment parameters
host_variants = ["qemu_kvm"]
vta_clk_freq_opts = [400]

# Build experiment for all combinations of parameters
for host_var, vta_clk_freq in itertools.product(
    host_variants, vta_clk_freq_opts
):
    experiment = Experiment(f"detect_enso-{host_var}-{vta_clk_freq}")
    pci_device_id = "0000:03:00.0"
    sync = False
    if host_var == "qemu_kvm":
        HostClass = QemuHost
    elif host_var == "qemu_icount":
        HostClass = QemuIcountHost
        sync = True

    net = SwitchNet()
    net.sync = sync

    experiment.add_network(net)
    server_config = TvmVtaCpuEnsoServer(pci_device_id)
    client_config = TvmVtaCpuEnsoClient()

    server = add_server(
        experiment, HostClass, server_config, net, sync, vta_clk_freq
    )
    client = add_client(experiment, HostClass, client_config, net, sync)

    server.wait = True  # Remove this when introducing client side.

    experiments.append(experiment)
