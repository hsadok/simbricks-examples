import enum
import os
import tarfile

from typing import Optional

import simbricks.orchestration.nodeconfig as node
import simbricks.orchestration.simulators as sim


#######################################
# Application configurations
# -------------------------------------


class TvmDeviceType(enum.Enum):
    VTA = "vta"
    CPU = "cpu"


#######################################
# Simulators
# -------------------------------------


class VTADev(sim.PCIDevSim):

    def __init__(self) -> None:
        super().__init__()
        self.clock_freq = 100
        """Clock frequency in MHz"""

    def run_cmd(self, env):
        cmd = (
            "./tvm-simbricks/3rdparty/vta-hw/simbricks/vta_simbricks "
            f"{env.dev_pci_path(self)} {env.dev_shm_path(self)} "
            f"{self.start_tick} {self.sync_period} {self.pci_latency} "
            f"{self.clock_freq}"
        )
        return cmd


class VtaNode(node.NodeConfig):
    def __init__(self) -> None:
        super().__init__()
        # Use locally built disk image
        self.disk_image = os.path.abspath("./output-tvm/tvm")
        # Bump amount of system memory
        self.memory = 3 * 1024
        # Reserve physical range of memory for the VTA user-space driver
        self.kcmd_append = " memmap=512M!1G"

        self.tvm_parent_dir = "/root"
        self.local_tvm_simbricks_dir = os.path.abspath("./tvm-simbricks")
        self.vm_mount_name = "tvm"

    def prepare_pre_cp(self) -> list[str]:
        # Define commands to run before application to configure the server
        cmds = super().prepare_pre_cp()
        if self.local_tvm_simbricks_dir is not None:
            cmds.extend(
                [
                    "ls -al /tmp/guest",
                    f"mkdir -p /tmp/guest/{self.vm_mount_name}-extract",
                    (
                        f"tar xf /tmp/guest/{self.vm_mount_name}"
                        f" -C /tmp/guest/{self.vm_mount_name}-extract"
                    ),
                    # Rsync using checksum to avoid copying files that are
                    # not modified.
                    f"ls -al /tmp/guest/{self.vm_mount_name}-extract",
                    f"rsync -a --exclude='build' --checksum"
                    f"  /tmp/guest/{self.vm_mount_name}-extract/tvm/"
                    f"  {self.tvm_parent_dir}/{self.vm_mount_name}",
                    f"ls -al {self.tvm_parent_dir}/{self.vm_mount_name}",
                    f"cd {self.tvm_parent_dir}/{self.vm_mount_name}",
                    "cd build",
                    "make -j`nproc`",
                ]
            )

        cmds.extend(
            [
                "mount -t proc proc /proc",
                "mount -t sysfs sysfs /sys",
                "ls -al /root",
                # "lspci -vv",
                "lspci -tvv",
                
                # Make TVM's Python framework available
                "export PYTHONPATH=/root/tvm/python:${PYTHONPATH}",
                "export PYTHONPATH=/root/tvm/vta/python:${PYTHONPATH}",

                f"export TVM_HOME={self.tvm_parent_dir}/{self.vm_mount_name}",

                # Set up loopback interface so the TVM inference script can
                # connect to the RPC server
                "ip link set lo up",
                "ip addr add 127.0.0.1/8 dev lo",
                # Make VTA device available for control from user-space via
                # VFIO
                (
                    "echo 1"
                    " >/sys/module/vfio/parameters/enable_unsafe_noiommu_mode"
                ),
                'echo "dead beef" >/sys/bus/pci/drivers/vfio-pci/new_id',
            ]
        )

        return cmds

    # pylint: disable=consider-using-with
    def config_files(self, environment):
        files = super().config_files(environment)
        if self.local_tvm_simbricks_dir is not None:
            # Tar the directory to be copied to the VM.
            tar_path = f"/tmp/{self.vm_mount_name}.tar"
            with tarfile.open(tar_path, "w") as tar:
                tar.add(self.local_tvm_simbricks_dir, arcname="tvm")

            files[self.vm_mount_name] = open(tar_path, "rb")

        return files
