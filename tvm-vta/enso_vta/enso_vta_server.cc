/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <enso/helpers.h>
#include <enso/pipe.h>
#include <support/socket.h>
#include <support/utils.h>
#include <tvm/runtime/logging.h>
#include <tvm_runner.h>
#include <unistd.h>

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "enso_vta.h"

static const std::string kUsage =
    "Command line usage\n"
    "--model        - The folder containing tvm artifacts(mod.so, mod.param, "
    "mod.json) \n"
    "--device       - The target device to use {llvm, opencl, cpu, cuda, "
    "extdev, metal, rocm, vpi, oneapi}\n"
    "--input        - Numpy file for the model input (optional and we use "
    "random if not given)\n"
    "--output       - Numpy file name to dump the model output as numpy\n"
    "--dump-meta    - Dump model meta information\n"
    "--pre-compiled - The file name of a file where pre-compiled programs "
    "should be stored\n"
    "\n"
    "  Example\n"
    "  ./rtvm --model=keras-resnet50 --device=\"opencl\" --dump-meta\n"
    "  ./rtvm --model=keras-resnet50 --device=\"opencl\" --input input.npz "
    "--output=output.npz\n"
    "\n";

/*!
 * \brief Tool Arguments.
 * \arg model The tvm artifact to load & run
 * \arg device The target device to use {llvm, cl, ...etc.}
 * \arg input Numpy file for the model input
 * \arg output Numpy file name to dump the model output as numpy
 * \arg pre_compiled File name where pre-compiled programs should be stored
 */
struct ToolArgs {
  std::string model;
  std::string device;
  std::string input;
  std::string output;
  std::string pre_compiled;
  bool dump_meta{false};
};

/*!
 * \brief PrintArgs print the contents of ToolArgs
 * \param args ToolArgs structure
 */
void PrintArgs(const ToolArgs& args) {
  LOG(INFO) << "Model         = " << args.model;
  LOG(INFO) << "Device        = " << args.device;
  LOG(INFO) << "Input         = " << args.input;
  LOG(INFO) << "Output        = " << args.output;
  LOG(INFO) << "Pre-compiled  = " << args.pre_compiled;
  LOG(INFO) << "Dump Metadata = " << ((args.dump_meta) ? ("True") : ("False"));
}

/*!
 * \brief ParseCmdArgs parses the command line arguments.
 * \param argc arg counter
 * \param argv arg values
 * \param args the output structure which holds the parsed values
 */
void ParseCmdArgs(int argc, char* argv[], struct ToolArgs& args) {
  const std::string model = GetCmdOption(argc, argv, "--model=");
  if (!model.empty()) {
    args.model = model;
  } else {
    LOG(INFO) << kUsage;
    exit(0);
  }

  const std::string device = GetCmdOption(argc, argv, "--device=");
  if (!device.empty()) {
    args.device = device;
  } else {
    LOG(INFO) << kUsage;
    exit(0);
  }

  const std::string input = GetCmdOption(argc, argv, "--input=");
  if (!input.empty()) {
    args.input = input;
  }

  const std::string output = GetCmdOption(argc, argv, "--output=");
  if (!output.empty()) {
    args.output = output;
  }

  const std::string pmeta = GetCmdOption(argc, argv, "--dump-meta", true);
  if (!pmeta.empty()) {
    args.dump_meta = true;
  }

  args.pre_compiled = GetCmdOption(argc, argv, "--pre-compiled=");
}

/*!
 * \brief Loads and Executes the model on given Target.
 * \param args tool arguments
 * \return result of operation.
 */
int ExecuteModel(ToolArgs& args) {
  HandleCtrlC();

  auto runner = new tvm::runtime::TVMRunner(args.model, args.device);

  runner->Load();
  if (!args.pre_compiled.empty()) {
    runner->UsePreCompiledPrograms(args.pre_compiled);
  }

  tvm::runtime::TVMMetaInfo mInfo = runner->GetMetaInfo();

  if (args.dump_meta) runner->PrintMetaInfo();

  std::unordered_map<std::string, DLTensor*> output_data;

  std::unique_ptr<enso::Device> dev = enso::Device::Create();
  std::vector<enso::RxPipe*> rx_pipes;
  enso::TxPipe* tx_pipe;

  if (!dev) {
    std::cerr << "Problem creating device" << std::endl;
    exit(2);
  }

  queue_extra_processing extra_processing_config = {
      .signal = 2,
      .config_id = kQueueExtraProcessingConfigId,
      .enso_pipe_id = 0,
      .extra_processing_type = kExtraProcessingMessage,
      .pad = {},
  };

  uint32_t dst_ip = kBaseServerIpAddress;
  uint16_t dst_port = kServerBasePort;

  // Create an RX pipe for each input.
  for ([[maybe_unused]] auto& elem : mInfo.input_info) {
    enso::RxPipe* rx_pipe = dev->AllocateRxPipe();
    if (!rx_pipe) {
      std::cerr << "Problem creating RX pipe" << std::endl;
      exit(3);
    }
    rx_pipe->Bind(dst_port, 0, dst_ip, 0, kProtocol);
    ++dst_port;

    // Enable message processing in the pipe.
    extra_processing_config.enso_pipe_id = rx_pipe->id();
    dev->ApplyConfig(
        reinterpret_cast<enso::TxNotification*>(&extra_processing_config));

    rx_pipes.push_back(rx_pipe);
  }

  tx_pipe = dev->AllocateTxPipe();
  if (!tx_pipe) {
    std::cerr << "Problem creating TX pipe" << std::endl;
    exit(4);
  }

  for (auto& elem : mInfo.output_info) {
    auto ndarr = tvm::runtime::NDArray::Empty(
        elem.second.first, tvm::runtime::String2DLDataType(elem.second.second),
        DLDevice{tvm::runtime::GetTVMDevice(args.device), 0});
    DLTensor* dl_tensor = ndarr.operator->();
    output_data.insert({elem.first, dl_tensor});
    runner->SetOutput(elem.first, dl_tensor);
  }

  std::cerr << "Waiting for inputs" << std::endl;

  size_t received_inputs = 0;
  uint32_t expected_msg_id = 0;
  while (received_inputs < rx_pipes.size()) {
    for (auto& rx_pipe : rx_pipes) {
      auto batch = rx_pipe->RecvMessages<PeekMsgIterator>();
      for (auto msg : batch) {
        MsgHeader* msg_hdr = reinterpret_cast<MsgHeader*>(msg);
        std::cerr << *msg_hdr << std::endl;
        if (msg_hdr->msg_id != expected_msg_id) {
          break;
        }
        ++expected_msg_id;
        ++received_inputs;

        DLTensor* dl_tensor;
        char* name;
        DeserializeDlTensor(msg + sizeof(MsgHeader), msg_hdr->total_length,
                            &dl_tensor, &name, sizeof(MsgHeader));
        runner->SetInput(std::string(name), dl_tensor);
      }
      uint32_t batch_length = batch.processed_bytes();
      if (unlikely(batch_length == 0)) {
        rx_pipe->Prefetch();
        continue;
      }
      rx_pipe->ConfirmBytes(batch_length);
    }
  }

  runner->Run();

  // Send output back.
  uint32_t tx_msg_id = 0;
  for (auto& elem : mInfo.output_info) {
    std::vector<uint8_t> bytes = SerializeDlTensor(
        elem.first, output_data[elem.first], sizeof(MsgHeader));
    size_t output_size = bytes.size();
    if (output_size == 0 || output_size > kMsgMaxTotalLength) {
      LOG(INFO) << "Invalid output size: " << output_size;
      return -1;
    }

    uint32_t offset = 0;
    while (offset < output_size) {
      uint8_t* tx_buf = tx_pipe->AllocateBuf();
      size_t buf_used = Pkt::SegmentData(
          tx_msg_id, (uint32_t)output_size, kBaseServerIpAddress,
          kBaseClientIpAddress, kServerBasePort, kClientPort, tx_buf,
          tx_pipe->capacity(), bytes.data(), &offset);
      tx_pipe->SendAndFree((uint32_t)buf_used);
    }
    ++tx_msg_id;
  }

  for (auto& rx_pipe : rx_pipes) {
    rx_pipe->Clear();
  }

  delete runner;

  return 0;
}

int main(int argc, char* argv[]) {
  if (argc <= 1) {
    LOG(INFO) << kUsage;
    return 0;
  }

  ToolArgs args;
  ParseCmdArgs(argc, argv, args);
  PrintArgs(args);

  if (ExecuteModel(args)) {
    PrintArgs(args);
    LOG(INFO) << kUsage;
    return -1;
  }
  return 0;
}
