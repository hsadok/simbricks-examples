import os
import sys

sys.path.append(str(os.path.dirname(__file__)))

import simbricks.orchestration.experiments as exp
import simbricks.orchestration.simulators as sim
import simbricks.orchestration.nodeconfig as node
import simbricks.orchestration.experiment.experiment_environment as env
import exp_util
import itertools
import os

experiments = []

# Experiment parameters
# host_variants = ["qemu_kvm", "qemu_icount"]
host_variants = ["qemu_kvm"]
# inference_device_opts = [exp_util.TvmDeviceType.CPU, exp_util.TvmDeviceType.VTA]
inference_device_opts = [exp_util.TvmDeviceType.VTA]
# vta_clk_freq_opts = [100, 400]
vta_clk_freq_opts = [400]

# Build experiment for all combinations of parameters
for host_var, inference_device, vta_clk_freq in itertools.product(
    host_variants, inference_device_opts, vta_clk_freq_opts
):
    experiment = exp.Experiment(
        f"detect_simple-{inference_device.value}-{host_var}-{vta_clk_freq}"
    )
    # pci_device_id = "0000:00:02.0"
    pci_device_id = "0000:03:00.0"
    sync = False
    if host_var == "qemu_kvm":
        HostClass = sim.QemuHost
    elif host_var == "qemu_icount":
        HostClass = sim.QemuIcountHost
        sync = True

    #######################################################
    # Define application
    # -----------------------------------------------------

    class TvmDetectLocalCpp(node.AppConfig):
        """Runs inference for detection model locally, either on VTA or the CPU."""

        def __init__(self):
            super().__init__()
            self.pci_device_id = pci_device_id
            self.device = inference_device
            self.debug = False
            """Whether to dump inference result."""

        def run_cmds(self, node):
            # define commands to run on simulated server
            cmds = [
                "ls -al /root/darknet",
                "cd /root/tvm/build/",
                (
                    f"VTA_DEVICE={self.pci_device_id} "
                    "LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH "
                    # "gdb -batch -ex \"catch syscall exit_group\" -ex \"run\" "
                    # "-ex \"bt\" -ex \"list\" --args "
                    "gdb --batch -x /root/tvm/cmds.gdb --args "
                    "/root/tvm/build/rtvm --model=/root/darknet "
                    "--device=extdev --dump-meta --zero-copy --dry-run 0"
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

    class TvmDetectLocal(node.AppConfig):
        """Runs inference for detection model locally using C++ backend."""

        def __init__(self):
            super().__init__()
            self.pci_device_id = pci_device_id
            self.device = inference_device
            self.test_img = "person.jpg"
            self.repetitions = 1
            self.debug = True
            """Whether to dump inference result."""

        def config_files(self, environment: env.ExpEnv):
            # mount TVM inference script in simulated server under /tmp/guest
            return {
                "deploy_detection-infer.py": open(
                    "./tvm_deploy_detection-infer.py", "rb"
                )
            }

        def run_cmds(self, node):
            # define commands to run on simulated server
            cmds = [
                # start RPC server
                f"VTA_DEVICE={self.pci_device_id} python3 -m"
                " vta.exec.rpc_server &"
                # wait for RPC server to start
                "sleep 6",
                f"export VTA_RPC_HOST=127.0.0.1",
                f"export VTA_RPC_PORT=9091",
                # run inference
                (
                    "python3 /tmp/guest/deploy_detection-infer.py "
                    f"/root/darknet {self.device.value} {self.test_img} "
                    f"{self.repetitions} {int(self.debug)}"
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

    #######################################################
    # Define and connect all simulators
    # -----------------------------------------------------
    # Instantiate server
    server_cfg = exp_util.VtaNode()
    # server_cfg.app = TvmDetectLocal()
    server_cfg.app = TvmDetectLocalCpp()
    server = HostClass(server_cfg)
    # Whether to synchronize VTA and server
    server.sync = sync
    # Wait until server exits
    server.wait = True
    server.name = 'vta_server'

    pci_sw = sim.PCISwitchSim()
    pci_sw.name = 'vta_pcisw'

    # Instantiate and connect VTA PCIe-based accelerator to server
    vta = exp_util.VTADev()
    vta.clock_freq = vta_clk_freq
    pci_sw.add_pcidev(vta)
    server.add_pcidev(pci_sw)

    # Add both simulators to experiment
    experiment.add_host(server)
    experiment.add_pcidev(vta)
    experiment.add_pcidev(pci_sw)

    experiments.append(experiment)
