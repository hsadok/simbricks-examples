/*
 * Copyright (c) 2025, Carnegie Mellon University
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the disclaimer
 * below) provided that the following conditions are met:
 *
 *      * Redistributions of source code must retain the above copyright notice,
 *      this list of conditions and the following disclaimer.
 *
 *      * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *      * Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
 * THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
 * NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <arpa/inet.h>
#include <netinet/ether.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <tvm/runtime/logging.h>

#include <algorithm>
#include <csignal>
#include <cstdint>
#include <string>
#include <vector>

static constexpr uint32_t ip_addr(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
  return (((uint32_t)a) << 24) | (((uint32_t)b) << 16) | (((uint32_t)c) << 8) |
         ((uint32_t)d);
}

constexpr uint32_t kBaseServerIpAddress = ip_addr(192, 168, 0, 0);
constexpr uint32_t kServerBasePort = 80;
constexpr uint32_t kBaseClientIpAddress = ip_addr(192, 168, 1, 0);
constexpr uint32_t kClientPort = 8080;
constexpr uint32_t kProtocol = IPPROTO_UDP;
constexpr uint32_t kMsgMaxTotalLength = (32768 - 2) * 64;

struct MsgHeader {
  uint32_t msg_id;        // Unique message ID.
  uint32_t total_length;  // In bytes.
  uint32_t offset;        // In bytes.
  friend std::ostream& operator<<(std::ostream& os, const MsgHeader& obj) {
    os << "MsgHeader(msg_id=" << obj.msg_id
       << ", total_length=" << obj.total_length << ", offset=" << obj.offset
       << ")";
    return os;
  }
} __attribute__((__packed__));

constexpr uint16_t kMaxMsgPayloadSize =
    ((1500 - sizeof(iphdr) - sizeof(udphdr) - sizeof(MsgHeader)) / 64) * 64;
static_assert(kMaxMsgPayloadSize % 64 == 0,
              "kMaxMsgPayloadSize must be 64-byte aligned");
static_assert(sizeof(iphdr) == 20);

constexpr uint64_t kQueueExtraProcessingConfigId = 5;
constexpr uint32_t kExtraProcessingMessage = 1;

struct __attribute__((__packed__)) queue_extra_processing {
  uint64_t signal;
  uint64_t config_id;
  uint32_t enso_pipe_id;
  uint32_t extra_processing_type;
  uint8_t pad[40];
};

class PeekMsgIterator : public enso::MessageIteratorBase<PeekMsgIterator> {
 public:
  inline PeekMsgIterator(uint8_t* addr, int32_t message_limit,
                         enso::RxPipe::MessageBatch<PeekMsgIterator>* batch)
      : MessageIteratorBase(addr, message_limit, batch) {}

  inline uint8_t* GetNextMessage(uint8_t* current_message) {
    MsgHeader* header = reinterpret_cast<MsgHeader*>(current_message);
    uint8_t* next_message = current_message + header->total_length;
    return next_message;
  }

  constexpr void OnAdvanceMessage([[maybe_unused]] uint32_t nb_bytes) {}
};

struct Pkt {
  ether_header l2_hdr;
  iphdr l3_hdr;
  udphdr l4_hdr;
  MsgHeader msg_hdr;

  Pkt(uint32_t msg_id, uint32_t total_length, uint32_t offset, uint32_t src_ip,
      uint32_t dst_ip, uint16_t src_port, uint16_t dst_port) {
    assert(total_length <= kMsgMaxTotalLength);
    assert(total_length > offset);
    uint16_t max_payload_len = kMaxMsgPayloadSize;
    if (offset == 0) {
      // First packet, we count the MsgHeader as part of the aligned payload.
      max_payload_len -= (uint16_t)sizeof(MsgHeader);
    }
    uint16_t payload_len = (uint16_t)std::min(
        (uint32_t)max_payload_len, (uint32_t)(total_length - offset));

    memset(&l2_hdr, 0, sizeof(ether_header));
    memset(&l3_hdr, 0, sizeof(iphdr));
    memset(&l4_hdr, 0, sizeof(udphdr));

    // Source and Destination MAC should be different to avoid problems with the
    // learning switch.
    l2_hdr.ether_dhost[0] = 0x00;
    l2_hdr.ether_dhost[1] = 0x00;
    l2_hdr.ether_dhost[2] = 0x00;
    l2_hdr.ether_dhost[3] = 0x00;
    l2_hdr.ether_dhost[4] = 0x00;
    l2_hdr.ether_dhost[5] = 0x00;

    l2_hdr.ether_shost[0] = 0x00;
    l2_hdr.ether_shost[1] = 0x00;
    l2_hdr.ether_shost[2] = 0x00;
    l2_hdr.ether_shost[3] = 0x00;
    l2_hdr.ether_shost[4] = 0x00;
    l2_hdr.ether_shost[5] = 0x01;

    l2_hdr.ether_type = htons(ETHERTYPE_IP);
    l3_hdr.ihl = 5;
    l3_hdr.version = 4;
    l3_hdr.ttl = 255;
    l3_hdr.protocol = IPPROTO_UDP;

    l3_hdr.saddr = htonl(src_ip);
    l3_hdr.daddr = htonl(dst_ip);
    l3_hdr.tot_len =
        htons(sizeof(l3_hdr) + sizeof(l4_hdr) + sizeof(msg_hdr) + payload_len);

    l4_hdr.source = htons(src_port);
    l4_hdr.dest = htons(dst_port);
    l4_hdr.len = htons(sizeof(l4_hdr) + sizeof(msg_hdr) + payload_len);

    msg_hdr.msg_id = msg_id;
    msg_hdr.total_length = total_length;
    msg_hdr.offset = offset;
  }

  /**
   * @brief Segments a large data payload into multiple network packets
   *
   * This function takes a data buffer and segments it into multiple network
   * packets that can be transmitted over a network. It creates properly
   * formatted packets with Ethernet, IP, and UDP headers, along with a custom
   * message header.
   *
   * @param msg_id Unique identifier for the message being segmented
   * @param total_length Total size of the complete message in bytes
   * @param src_ip Source IP address for the packets
   * @param dst_ip Destination IP address for the packets
   * @param src_port Source UDP port
   * @param dst_port Destination UDP port
   * @param dst_buf Destination buffer where the formatted packets will be
   *                written
   * @param dst_buf_len Size of the destination buffer in bytes
   * @param src_data Pointer to the source data to be segmented
   * @param offset Pointer to the current offset in the message; will be updated
   *               after call.
   *
   * @return Size in bytes of data written to the destination buffer
   *
   * @note The function returns the total number of bytes copied to the buffer.
   *       It also updates the offset of the data that was copied. The sender
   *       should check the updated offset to determine if the entire message
   *       was copied. Otherwise, the sender should call the function again with
   *       the updated offset and more buffer space. That also means that the
   *       next call should provide a different dst_buf pointer that starts at
   *       the end of the previous buffer.
   */
  static size_t SegmentData(uint32_t msg_id, uint32_t total_length,
                            uint32_t src_ip, uint32_t dst_ip, uint16_t src_port,
                            uint16_t dst_port, uint8_t* dst_buf,
                            uint32_t dst_buf_len, const uint8_t* src_data,
                            uint32_t* offset) {
    size_t buf_used = 0;
    uint32_t offset_ = *offset;
    while (offset_ < total_length) {
      if (buf_used + sizeof(Pkt) > dst_buf_len) {
        break;
      }

      Pkt* pkt = new (dst_buf) Pkt(msg_id, total_length, offset_, src_ip,
                                   dst_ip, src_port, dst_port);
      uint32_t payload_len = pkt->payload_len();

      if (buf_used + sizeof(Pkt) + payload_len > dst_buf_len) {
        break;
      }

      memcpy(dst_buf + sizeof(Pkt), src_data, payload_len);

      src_data += payload_len;
      offset_ += payload_len;

      uint32_t aligned_len = (sizeof(Pkt) + payload_len + 63) & ~((uint32_t)63);
      dst_buf += aligned_len;
      buf_used += aligned_len;
    }

    *offset = offset_;
    return buf_used;
  }

  static uint32_t SegmentationOverhead(uint32_t total_length) {
    size_t nb_packets =
        (total_length + sizeof(MsgHeader) + kMaxMsgPayloadSize - 1) /
        kMaxMsgPayloadSize;
    size_t overhead = nb_packets * sizeof(Pkt);

    size_t first_payload_len =
        std::min((size_t)total_length, kMaxMsgPayloadSize - sizeof(MsgHeader));
    size_t first_pkt_len = first_payload_len + sizeof(Pkt);
    overhead += ((first_pkt_len + 63) & ~63lu) - first_pkt_len;

    if (nb_packets == 1) {
      return (uint32_t)overhead;
    }

    size_t last_payload_len =
        (total_length - first_payload_len) % kMaxMsgPayloadSize;
    size_t last_pkt_len = last_payload_len + sizeof(Pkt);
    overhead += ((last_pkt_len + 63) & ~63lu) - last_pkt_len;

    if (nb_packets == 2) {
      return (uint32_t)overhead;
    }

    size_t middle_pkt_len = kMaxMsgPayloadSize + sizeof(Pkt);
    size_t middle_pkt_overhead =
        ((middle_pkt_len + 63) & ~63lu) - middle_pkt_len;

    overhead += (middle_pkt_overhead) * (nb_packets - 2);
    return (uint32_t)overhead;
  }

  inline uint16_t payload_len() const {
    return ntohs(l3_hdr.tot_len) - (uint16_t)sizeof(l3_hdr) -
           (uint16_t)sizeof(l4_hdr) - (uint16_t)sizeof(msg_hdr);
  }
} __attribute__((__packed__));
static_assert(sizeof(Pkt) == sizeof(ether_header) + sizeof(iphdr) +
                                 sizeof(udphdr) + sizeof(MsgHeader));

/*!
 * \brief CtrlCHandler, exits if Ctrl+C is pressed
 * \param s signal
 */
void CtrlCHandler([[maybe_unused]] int s) {
  LOG(INFO) << "\nUser pressed Ctrl+C, Exiting";
  exit(1);
}

/*!
 * \brief HandleCtrlC Register for handling Ctrl+C event.
 */
void HandleCtrlC() {
  // Ctrl+C handler
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = CtrlCHandler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, nullptr);
}

/*!
 * \brief GetCmdOption Parse and find the command option.
 * \param argc arg counter
 * \param argv arg values
 * \param option command line option to search for.
 * \param key whether the option itself is key
 * \return value corresponding to option.
 */
std::string GetCmdOption(int argc, char* argv[], std::string option,
                         bool key = false) {
  std::string cmd;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find(option) == 0) {
      if (key) {
        cmd = argv[i];
        return cmd;
      }
      // We assume "=" is the end of option.
      ICHECK_EQ(*option.rbegin(), '=');
      cmd = arg.substr(arg.find('=') + 1);
      return cmd;
    }
  }
  return cmd;
}

// Extracted from dlpack.h docstring.
inline size_t GetDataSize(const DLTensor* t) {
  size_t size = 1;
  for (tvm_index_t i = 0; i < t->ndim; ++i) {
    size *= (size_t)t->shape[i];
  }
  size *= ((size_t)t->dtype.bits * t->dtype.lanes + 7) / 8;
  return size;
}

// Data must be aligned to 64B in the destination buffer. Since the alignment
// depends on the header, the caller should provide `data_alignment_offset`
// with the offset caused by the header in the destination buffer.
inline std::vector<uint8_t> SerializeDlTensor(const std::string& name,
                                              const DLTensor* dl_tensor,
                                              size_t data_alignment_offset) {
  std::vector<uint8_t> bytes;
  std::cerr << "Serializing tensor" << std::endl;
  // We place the data such that the string name is at front. We can detect
  // the beginning of the data by scanning for the terminating byte: `\0`.
  bytes.insert(bytes.end(), name.begin(), name.end());
  bytes.push_back('\0');
  std::cerr << "name: \"" << name << "\"" << std::endl;
  bytes.insert(bytes.end(), reinterpret_cast<const uint8_t*>(dl_tensor),
               reinterpret_cast<const uint8_t*>(dl_tensor) + sizeof(DLTensor));
  std::cerr << "DLTensor: " << std::endl;
  std::cerr << "  data: " << dl_tensor->data << std::endl;
  std::cerr << "  device.device_type: " << dl_tensor->device.device_type
            << std::endl;
  std::cerr << "  device.device_id: " << dl_tensor->device.device_id
            << std::endl;
  std::cerr << "  ndim: " << dl_tensor->ndim << std::endl;
  std::cerr << "  dtype.code: " << (int)(dl_tensor->dtype.code) << std::endl;
  std::cerr << "  dtype.bits: " << (int)(dl_tensor->dtype.bits) << std::endl;
  std::cerr << "  dtype.lanes: " << dl_tensor->dtype.lanes << std::endl;
  std::cerr << "  shape: " << (void*)(dl_tensor->shape) << std::endl;
  std::cerr << "  strides: " << (void*)(dl_tensor->strides) << std::endl;
  std::cerr << "  byte_offset: " << dl_tensor->byte_offset << std::endl;

  if (dl_tensor->shape != nullptr) {
    bytes.insert(bytes.end(), reinterpret_cast<uint8_t*>(dl_tensor->shape),
                 reinterpret_cast<uint8_t*>(dl_tensor->shape) +
                     sizeof(*(dl_tensor->shape)) * (size_t)dl_tensor->ndim);
  }
  if (dl_tensor->data != nullptr) {
    assert(dl_tensor->shape != nullptr);
    size_t align = (64 - ((bytes.size() + data_alignment_offset) & 63)) & 63;
    std::cerr << "Data offset: " << align << std::endl;
    bytes.insert(bytes.end(), align, 0);

    bytes.insert(bytes.end(), reinterpret_cast<const uint8_t*>(dl_tensor->data),
                 reinterpret_cast<const uint8_t*>(dl_tensor->data) +
                     GetDataSize(dl_tensor));
  }
  if (dl_tensor->strides != nullptr) {
    bytes.insert(bytes.end(), reinterpret_cast<uint8_t*>(dl_tensor->strides),
                 reinterpret_cast<uint8_t*>(dl_tensor->strides) +
                     sizeof(*(dl_tensor->strides)) * (size_t)dl_tensor->ndim);
  }

  std::cerr << "Total size: " << bytes.size() << std::endl;
  return bytes;
}

inline void DeserializeDlTensor(uint8_t* serialized_data, size_t length,
                                DLTensor** dl_tensor, char** name,
                                size_t data_alignment_offset) {
  assert(data_alignment_offset == (size_t)serialized_data & 63);
  size_t offset = 0;
  size_t name_end = 0;
  while (name_end < length && serialized_data[name_end] != '\0') {
    ++name_end;
  }
  assert(name_end < length);
  *name = reinterpret_cast<char*>(serialized_data);
  offset = name_end + 1;

  std::cerr << "name: \"" << *name << "\"" << std::endl;

  *dl_tensor = reinterpret_cast<DLTensor*>(serialized_data + offset);
  offset += sizeof(DLTensor);

  assert(offset < length);

  std::cerr << "DLTensor: " << std::endl;
  std::cerr << "  data: " << (*dl_tensor)->data << std::endl;
  std::cerr << "  device.device_type: " << (*dl_tensor)->device.device_type
            << std::endl;
  std::cerr << "  device.device_id: " << (*dl_tensor)->device.device_id
            << std::endl;
  std::cerr << "  ndim: " << (*dl_tensor)->ndim << std::endl;
  std::cerr << "  dtype.code: " << (int)((*dl_tensor)->dtype.code) << std::endl;
  std::cerr << "  dtype.bits: " << (int)((*dl_tensor)->dtype.bits) << std::endl;
  std::cerr << "  dtype.lanes: " << (*dl_tensor)->dtype.lanes << std::endl;
  std::cerr << "  shape: " << (void*)((*dl_tensor)->shape) << std::endl;
  std::cerr << "  strides: " << (void*)(*dl_tensor)->strides << std::endl;
  std::cerr << "  byte_offset: " << (*dl_tensor)->byte_offset << std::endl;

  if ((*dl_tensor)->shape != nullptr) {
    (*dl_tensor)->shape =
        reinterpret_cast<tvm_index_t*>(serialized_data + offset);
    offset += sizeof(tvm_index_t) * (size_t)(*dl_tensor)->ndim;
  }

  assert(offset < length);

  if ((*dl_tensor)->data != nullptr) {
    size_t align = (64 - ((offset + data_alignment_offset) & 63)) & 63;
    offset += align;
    (*dl_tensor)->data = reinterpret_cast<void*>(serialized_data + offset);
    offset += GetDataSize(*dl_tensor);
  }

  assert(offset < length);

  if ((*dl_tensor)->strides != nullptr) {
    (*dl_tensor)->strides =
        reinterpret_cast<tvm_index_t*>(serialized_data + offset);
    offset += sizeof(tvm_index_t) * (size_t)(*dl_tensor)->ndim;
  }

  std::cerr << "Total size: " << offset << std::endl;
  assert(offset == length);
}
