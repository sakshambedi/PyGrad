// Copyright 2025 Saksham Bedi hello@sakshambedi.com
// All rights reserved.
#include "../../kernels/cpu_kernel.h"
#include "../../kernels/operations.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
TEST(BufferTest, ConstructorAndSize) {
  py::scoped_interpreter guard{};

  Buffer fbuf(5, "float32");
  EXPECT_EQ(fbuf.size(), 5);
  EXPECT_EQ(fbuf.dtype(), "float32");

  Buffer ibuf(3, "int64");
  EXPECT_EQ(ibuf.size(), 3);
  EXPECT_EQ(ibuf.dtype(), "int64");

  Buffer bbuf(2, "uint8");
  EXPECT_EQ(bbuf.size(), 2);
  EXPECT_EQ(bbuf.dtype(), "uint8");
}

TEST(FilledBufferTest, FilledIntegerBuffers) {
  py::scoped_interpreter guard{};

  py::object int_val = py::cast(42);

  Buffer int8_buf = Buffer::_filled(int_val, "int8", 5);
  EXPECT_EQ(int8_buf.size(), 5);
  EXPECT_EQ(int8_buf.dtype(), "int8");

  for (size_t i = 0; i < int8_buf.size(); ++i) {
    EXPECT_EQ(int8_buf.get_item(i).cast<int8_t>(), 42);
  }

  Buffer int16_buf = Buffer::_filled(int_val, "int16", 10);
  EXPECT_EQ(int16_buf.size(), 10);
  EXPECT_EQ(int16_buf.dtype(), "int16");

  for (size_t i = 0; i < int16_buf.size(); ++i) {
    EXPECT_EQ(int16_buf.get_item(i).cast<int16_t>(), 42);
  }

  Buffer int32_buf = Buffer::_filled(int_val, "int32", 3);
  EXPECT_EQ(int32_buf.size(), 3);
  EXPECT_EQ(int32_buf.dtype(), "int32");

  for (size_t i = 0; i < int32_buf.size(); ++i) {
    EXPECT_EQ(int32_buf.get_item(i).cast<int32_t>(), 42);
  }

  Buffer int64_buf = Buffer::_filled(int_val, "int64", 7);
  EXPECT_EQ(int64_buf.size(), 7);
  EXPECT_EQ(int64_buf.dtype(), "int64");

  for (size_t i = 0; i < int64_buf.size(); ++i) {
    EXPECT_EQ(int64_buf.get_item(i).cast<int64_t>(), 42);
  }
}

TEST(FilledBufferTest, FilledUnsignedIntegerBuffers) {
  py::scoped_interpreter guard{};

  py::object uint_val = py::cast(255);

  Buffer uint8_buf = Buffer::_filled(uint_val, "uint8", 6);
  EXPECT_EQ(uint8_buf.size(), 6);
  EXPECT_EQ(uint8_buf.dtype(), "uint8");

  for (size_t i = 0; i < uint8_buf.size(); ++i) {
    EXPECT_EQ(uint8_buf.get_item(i).cast<uint8_t>(), 255);
  }

  Buffer uint16_buf = Buffer::_filled(uint_val, "uint16", 4);
  EXPECT_EQ(uint16_buf.size(), 4);
  EXPECT_EQ(uint16_buf.dtype(), "uint16");

  for (size_t i = 0; i < uint16_buf.size(); ++i) {
    EXPECT_EQ(uint16_buf.get_item(i).cast<uint16_t>(), 255);
  }

  Buffer uint32_buf = Buffer::_filled(uint_val, "uint32", 8);
  EXPECT_EQ(uint32_buf.size(), 8);
  EXPECT_EQ(uint32_buf.dtype(), "uint32");

  for (size_t i = 0; i < uint32_buf.size(); ++i) {
    EXPECT_EQ(uint32_buf.get_item(i).cast<uint32_t>(), 255);
  }

  Buffer uint64_buf = Buffer::_filled(uint_val, "uint64", 2);
  EXPECT_EQ(uint64_buf.size(), 2);
  EXPECT_EQ(uint64_buf.dtype(), "uint64");

  for (size_t i = 0; i < uint64_buf.size(); ++i) {
    EXPECT_EQ(uint64_buf.get_item(i).cast<uint64_t>(), 255);
  }
}

TEST(FilledBufferTest, FilledFloatBuffers) {
  py::scoped_interpreter guard{};

  py::object float_val = py::cast(3.14159f);

  Buffer float32_buf = Buffer::_filled(float_val, "float32", 9);
  EXPECT_EQ(float32_buf.size(), 9);
  EXPECT_EQ(float32_buf.dtype(), "float32");

  for (size_t i = 0; i < float32_buf.size(); ++i) {
    EXPECT_NEAR(float32_buf.get_item(i).cast<float>(), 3.14159f, 1e-5f);
  }

  Buffer float64_buf = Buffer::_filled(float_val, "float64", 4);
  EXPECT_EQ(float64_buf.size(), 4);
  EXPECT_EQ(float64_buf.dtype(), "float64");

  for (size_t i = 0; i < float64_buf.size(); ++i) {
    EXPECT_NEAR(float64_buf.get_item(i).cast<double>(), 3.14159, 1e-5);
  }

  py::object int_val = py::cast(42);
  Buffer int_to_float_buf = Buffer::_filled(int_val, "float32", 3);
  EXPECT_EQ(int_to_float_buf.size(), 3);
  EXPECT_EQ(int_to_float_buf.dtype(), "float32");

  for (size_t i = 0; i < int_to_float_buf.size(); ++i) {
    EXPECT_FLOAT_EQ(int_to_float_buf.get_item(i).cast<float>(), 42.0f);
  }
}

TEST(FilledBufferTest, EdgeCases) {
  py::scoped_interpreter guard{};

  // Empty buffer - Verify that a buffer can be created with zero size.
  py::object val = py::cast(1);
  Buffer empty_buf = Buffer::_filled(val, "int32", 0);
  EXPECT_EQ(empty_buf.size(), 0);
  EXPECT_EQ(empty_buf.dtype(), "int32");

  // Large buffer - Check memory allocation and basic access for a large buffer.
  const size_t large_size = 1000000;
  Buffer large_buf = Buffer::_filled(val, "int32", large_size);
  EXPECT_EQ(large_buf.size(), large_size);
  EXPECT_EQ(large_buf.dtype(), "int32");

  EXPECT_EQ(large_buf.get_item(0).cast<int32_t>(), 1);
  EXPECT_EQ(large_buf.get_item(large_size / 2).cast<int32_t>(), 1);
  EXPECT_EQ(large_buf.get_item(large_size - 1).cast<int32_t>(), 1);

  // Special numeric values - Ensure correct handling of 0 and negative numbers.
  py::object zero_val = py::cast(0);
  Buffer zero_buf = Buffer::_filled(zero_val, "float32", 10);
  for (size_t i = 0; i < zero_buf.size(); ++i) {
    EXPECT_EQ(zero_buf.get_item(i).cast<float>(), 0.0f);
  }

  py::object neg_val = py::cast(-1);
  Buffer neg_buf = Buffer::_filled(neg_val, "int32", 10);
  for (size_t i = 0; i < neg_buf.size(); ++i) {
    EXPECT_EQ(neg_buf.get_item(i).cast<int32_t>(), -1);
  }

  // Extreme values - Test with the maximum and minimum values of a data type.
  py::object max_int_val = py::cast(std::numeric_limits<int32_t>::max());
  Buffer max_int_buf = Buffer::_filled(max_int_val, "int32", 5);
  for (size_t i = 0; i < max_int_buf.size(); ++i) {
    EXPECT_EQ(max_int_buf.get_item(i).cast<int32_t>(),
              std::numeric_limits<int32_t>::max());
  }

  py::object min_int_val = py::cast(std::numeric_limits<int32_t>::min());
  Buffer min_int_buf = Buffer::_filled(min_int_val, "int32", 5);
  for (size_t i = 0; i < min_int_buf.size(); ++i) {
    EXPECT_EQ(min_int_buf.get_item(i).cast<int32_t>(),
              std::numeric_limits<int32_t>::min());
  }
}

TEST(FilledBufferTest, TypeConversions) {
  py::scoped_interpreter guard{};

  py::object int_val = py::cast(42);

  Buffer float32_buf = Buffer::_filled(int_val, "float32", 3);
  EXPECT_EQ(float32_buf.dtype(), "float32");
  EXPECT_EQ(float32_buf.size(), 3);

  for (size_t i = 0; i < float32_buf.size(); ++i) {
    EXPECT_FLOAT_EQ(float32_buf.get_item(i).cast<float>(), 42.0f);
  }

  Buffer float64_buf = Buffer::_filled(int_val, "float64", 3);
  EXPECT_EQ(float64_buf.dtype(), "float64");
  EXPECT_EQ(float64_buf.size(), 3);

  for (size_t i = 0; i < float64_buf.size(); ++i) {
    EXPECT_DOUBLE_EQ(float64_buf.get_item(i).cast<double>(), 42.0);
  }

  py::object small_int = py::cast(127);

  Buffer int8_buf = Buffer::_filled(small_int, "int8", 3);
  EXPECT_EQ(int8_buf.dtype(), "int8");

  for (size_t i = 0; i < int8_buf.size(); ++i) {
    EXPECT_EQ(int8_buf.get_item(i).cast<int8_t>(), 127);
  }

  Buffer uint8_buf = Buffer::_filled(small_int, "uint8", 3);
  EXPECT_EQ(uint8_buf.dtype(), "uint8");

  for (size_t i = 0; i < uint8_buf.size(); ++i) {
    EXPECT_EQ(uint8_buf.get_item(i).cast<uint8_t>(), 127);
  }
}

TEST(FilledBufferTest, FullMethodAlias) {
  py::scoped_interpreter guard{};

  py::object val = py::cast(7);

  Buffer buf1 = Buffer::_filled(val, "int16", 5);
  Buffer buf2 = Buffer::_filled(val, "int16", 5);

  EXPECT_EQ(buf1.size(), buf2.size());
  EXPECT_EQ(buf1.dtype(), buf2.dtype());

  for (size_t i = 0; i < buf1.size(); ++i) {
    EXPECT_EQ(buf1.get_item(i).cast<int16_t>(),
              buf2.get_item(i).cast<int16_t>());
  }
}

TEST(FilledBufferTest, SpecialFloatValues) {
  py::scoped_interpreter guard{};

  // NaN values - Ensure NaN is correctly propagated in float buffers.
  py::object nan_val = py::eval("float('nan')");
  Buffer nan_buf = Buffer::_filled(nan_val, "float32", 5);
  EXPECT_EQ(nan_buf.size(), 5);
  EXPECT_EQ(nan_buf.dtype(), "float32");

  for (size_t i = 0; i < nan_buf.size(); ++i) {
    auto value = nan_buf.get_item(i).cast<float>();
    EXPECT_TRUE(std::isnan(value));
  }

  // Infinity values - Ensure infinity is correctly propagated.
  py::object inf_val = py::eval("float('inf')");
  Buffer inf_buf = Buffer::_filled(inf_val, "float32", 5);
  EXPECT_EQ(inf_buf.size(), 5);
  EXPECT_EQ(inf_buf.dtype(), "float32");

  for (size_t i = 0; i < inf_buf.size(); ++i) {
    auto value = inf_buf.get_item(i).cast<float>();
    EXPECT_TRUE(std::isinf(value));
  }

  // Negative infinity - Ensure negative infinity is correctly propagated.
  py::object neg_inf_val = py::eval("float('-inf')");
  Buffer neg_inf_buf = Buffer::_filled(neg_inf_val, "float64", 3);
  EXPECT_EQ(neg_inf_buf.size(), 3);
  EXPECT_EQ(neg_inf_buf.dtype(), "float64");

  for (size_t i = 0; i < neg_inf_buf.size(); ++i) {
    auto value = neg_inf_buf.get_item(i).cast<double>();
    EXPECT_TRUE(std::isinf(value));
    EXPECT_LT(value, 0.0);
  }
}

TEST(FilledBufferTest, ExtremeSizes) {
  py::scoped_interpreter guard{};

  // Single element buffer - A common edge case.
  py::object val = py::cast(42);
  Buffer single_buf = Buffer::_filled(val, "int32", 1);
  EXPECT_EQ(single_buf.size(), 1);
  EXPECT_EQ(single_buf.get_item(0).cast<int32_t>(), 42);

  // Small non-zero buffer - Test sizes that are not multiples of SIMD width.
  Buffer small_buf = Buffer::_filled(val, "int32", 2);
  EXPECT_EQ(small_buf.size(), 2);
  EXPECT_EQ(small_buf.get_item(0).cast<int32_t>(), 42);
  EXPECT_EQ(small_buf.get_item(1).cast<int32_t>(), 42);
}

TEST(FilledBufferTest, ErrorConditions) {
  py::scoped_interpreter guard{};

  // Invalid dtype string - The constructor must throw an error for unknown
  // types.
  py::object val = py::cast(42);

  EXPECT_THROW(
      { Buffer buf = Buffer::_filled(val, "invalid_dtype", 5); },
      std::runtime_error);

  EXPECT_THROW(
      { Buffer buf = Buffer::_filled(val, "", 5); }, std::runtime_error);

  EXPECT_THROW(
      { Buffer buf = Buffer::_filled(val, "not_a_valid_type", 100); },
      std::runtime_error);
}

TEST(BufferTest, DTypeCheck) {
  py::scoped_interpreter guard{};

  Buffer f16buf(1, "float16");
  EXPECT_EQ(f16buf.dtype(), "float16");

  Buffer f32buf(1, "float32");
  EXPECT_EQ(f32buf.dtype(), "float32");

  Buffer f64buf(1, "float64");
  EXPECT_EQ(f64buf.dtype(), "float64");

  Buffer i8buf(1, "int8");
  EXPECT_EQ(i8buf.dtype(), "int8");

  Buffer u8buf(1, "uint8");
  EXPECT_EQ(u8buf.dtype(), "uint8");

  Buffer i16buf(1, "int16");
  EXPECT_EQ(i16buf.dtype(), "int16");

  Buffer u16buf(1, "uint16");
  EXPECT_EQ(u16buf.dtype(), "uint16");

  Buffer i32buf(1, "int32");
  EXPECT_EQ(i32buf.dtype(), "int32");

  Buffer u32buf(1, "uint32");
  EXPECT_EQ(u32buf.dtype(), "uint32");

  Buffer i64buf(1, "int64");
  EXPECT_EQ(i64buf.dtype(), "int64");

  Buffer u64buf(1, "uint64");
  EXPECT_EQ(u64buf.dtype(), "uint64");
}

// Test set and get operations for various dtypes
// TEST(BufferTest, SetAndGet) {
//   py::scoped_interpreter guard{};
//
//   Buffer fbuf(1, "float32");
//   fbuf.set_item(0, 3.5);
//   EXPECT_NEAR(py::cast<float>(fbuf.get_item(0)), 3.5f, 1e-6f);
//
//   Buffer ibuf(1, "int32");
//   ibuf.set_item(0, 7);
//   EXPECT_EQ(py::cast<int>(ibuf.get_item(0)), 7);
// }

TEST(BufferTest, InitializerListConstructor) {
  py::scoped_interpreter guard{};

  Buffer fbuf({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, "float32");
  EXPECT_EQ(fbuf.size(), 5);
  EXPECT_EQ(fbuf.dtype(), "float32");
  EXPECT_NEAR(py::cast<float>(fbuf.get_item(0)), 1.0f, 1e-6f);
  EXPECT_NEAR(py::cast<float>(fbuf.get_item(1)), 2.0f, 1e-6f);
  EXPECT_NEAR(py::cast<float>(fbuf.get_item(2)), 3.0f, 1e-6f);
  EXPECT_NEAR(py::cast<float>(fbuf.get_item(3)), 4.0f, 1e-6f);
  EXPECT_NEAR(py::cast<float>(fbuf.get_item(4)), 5.0f, 1e-6f);

  Buffer ibuf({-1, 0, 1}, "int64");
  EXPECT_EQ(ibuf.size(), 3);
  EXPECT_EQ(ibuf.dtype(), "int64");
  EXPECT_EQ(py::cast<int64_t>(ibuf.get_item(0)), -1);
  EXPECT_EQ(py::cast<int64_t>(ibuf.get_item(1)), 0);
  EXPECT_EQ(py::cast<int64_t>(ibuf.get_item(2)), 1);

  Buffer u8buf({0, 128, 255}, "uint8");
  EXPECT_EQ(u8buf.size(), 3);
  EXPECT_EQ(u8buf.dtype(), "uint8");
  EXPECT_EQ(py::cast<uint8_t>(u8buf.get_item(0)), 0);
  EXPECT_EQ(py::cast<uint8_t>(u8buf.get_item(1)), 128);
  EXPECT_EQ(py::cast<uint8_t>(u8buf.get_item(2)), 255);

  Buffer dbuf({1.5, 2.5, 3.14159}, "float64");
  EXPECT_EQ(dbuf.size(), 3);
  EXPECT_EQ(dbuf.dtype(), "float64");
  EXPECT_NEAR(py::cast<double>(dbuf.get_item(0)), 1.5, 1e-9);
  EXPECT_NEAR(py::cast<double>(dbuf.get_item(1)), 2.5, 1e-9);
  EXPECT_NEAR(py::cast<double>(dbuf.get_item(2)), 3.14159, 1e-9);

  Buffer empty_buf({}, "int16");
  EXPECT_EQ(empty_buf.size(), 0);
  EXPECT_EQ(empty_buf.dtype(), "int16");

  Buffer conv_buf({1.1f, 2.9f, 3.5f}, "int32");
  EXPECT_EQ(conv_buf.size(), 3);
  EXPECT_EQ(conv_buf.dtype(), "int32");

  EXPECT_EQ(py::cast<int32_t>(conv_buf.get_item(0)), 1);
  EXPECT_EQ(py::cast<int32_t>(conv_buf.get_item(1)), 2);
  EXPECT_EQ(py::cast<int32_t>(conv_buf.get_item(2)), 3);
}

TEST(BufferTest, PyBufferConstructor) {
  py::scoped_interpreter guard{};

  py::module np = py::module::import("numpy");

  py::array_t<float> np_float =
      np.attr("array")(py::make_tuple(1.0f, 2.0f, 3.0f, 4.0f), "float32");
  Buffer float_buf(np_float, "float32");
  EXPECT_EQ(float_buf.size(), 4);
  EXPECT_EQ(float_buf.dtype(), "float32");

  py::array_t<int32_t> np_int =
      np.attr("array")(py::make_tuple(10, 20, 30), "int32");
  Buffer int_buf(np_int, "int32");
  EXPECT_EQ(int_buf.size(), 3);
  EXPECT_EQ(int_buf.dtype(), "int32");

  py::array_t<double> np_empty = np.attr("array")(py::make_tuple(), "float64");
  Buffer empty_buf(np_empty, "float64");
  EXPECT_EQ(empty_buf.size(), 0);
  EXPECT_EQ(empty_buf.dtype(), "float64");

  py::array_t<int> np_mix =
      np.attr("array")(py::make_tuple(0, 128, 255), "int32");
  Buffer uint_buf(np_mix, "uint8");
  EXPECT_EQ(uint_buf.size(), 3);
  EXPECT_EQ(uint_buf.dtype(), "uint8");
}

TEST(BufferTest, ConstructorExceptions) {
  py::scoped_interpreter guard{};

  EXPECT_THROW(Buffer(5, "invalid_type"), std::runtime_error);
  EXPECT_THROW(Buffer({1, 2, 3}, "float128"), std::runtime_error);
}

// Out-of-bounds access - Accessing an index beyond the buffer size must throw.
TEST(BufferTest, GetItemOutOfBounds) {
  py::scoped_interpreter guard{};
  Buffer buf(1, "float32");
  EXPECT_THROW(
      {
        auto result = buf.get_item(2);
        static_cast<void>(result);
      },
      std::out_of_range);
}

TEST(CastBufferTest, Int32ToFloat32) {
  VecBuffer<int32_t> in({10, -5, 0, 123});
  auto out = in.cast<float>();
  ASSERT_EQ(out.size(), in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(in[i]), out[i]);
  }
}

TEST(CastBufferTest, Float32ToInt16) {
  VecBuffer<float> in({1.9f, -2.1f, 0.0f, 32767.0f});
  auto out = in.cast<int16_t>();
  ASSERT_EQ(out.size(), in.size());
  EXPECT_EQ(out[0], static_cast<int16_t>(1.9f));
  EXPECT_EQ(out[1], static_cast<int16_t>(-2.1f));
  EXPECT_EQ(out[2], static_cast<int16_t>(0.0f));
  EXPECT_EQ(out[3], static_cast<int16_t>(32767.0f));
}

TEST(CastBufferTest, UInt8ToFloat64) {
  VecBuffer<uint8_t> in({0, 128, 255});
  auto out = in.cast<double>();
  ASSERT_EQ(out.size(), in.size());
  EXPECT_DOUBLE_EQ(static_cast<double>(in[0]), out[0]);
  EXPECT_DOUBLE_EQ(static_cast<double>(in[1]), out[1]);
  EXPECT_DOUBLE_EQ(static_cast<double>(in[2]), out[2]);
}

TEST(CastBufferTest, Float64ToInt64) {
  VecBuffer<double> in(
      {1.1, -2.9, 0.0, 9223372036854774000.0, -9223372036854774000.0});
  auto out = in.cast<int64_t>();
  ASSERT_EQ(out.size(), in.size());
  EXPECT_EQ(out[0], static_cast<int64_t>(1.1));
  EXPECT_EQ(out[1], static_cast<int64_t>(-2.9));
  EXPECT_EQ(out[2], static_cast<int64_t>(0.0));
  EXPECT_EQ(out[3], static_cast<int64_t>(9223372036854774000.0));
  EXPECT_EQ(out[4], static_cast<int64_t>(-9223372036854774000.0));
}

TEST(CastBufferTest, Int16ToFloat16) {
  VecBuffer<int16_t> in({-32768, -1, 0, 1, 32767});
  auto out = in.cast<half>();
  ASSERT_EQ(out.size(), in.size());

  std::array<float, 5> values;
  for (size_t i = 0; i < out.size(); ++i) {
    values[i] = static_cast<float>(out[i]);
  }

  // Allow some precision loss for float16 conversion.
  EXPECT_NEAR(values[0], static_cast<float>(-32768), 8.0f);
  EXPECT_NEAR(values[1], -1.0f, 0.01f);
  EXPECT_NEAR(values[2], 0.0f, 0.01f);
  EXPECT_NEAR(values[3], 1.0f, 0.01f);
  EXPECT_NEAR(values[4], static_cast<float>(32767), 8.0f);
}

TEST(CastBufferTest, UInt32ToInt8) {
  VecBuffer<uint32_t> in({0, 127, 128, 255, 256, 1000000});
  auto out = in.cast<int8_t>();
  ASSERT_EQ(out.size(), in.size());

  EXPECT_EQ(out[0], static_cast<int8_t>(0));
  EXPECT_EQ(out[1], static_cast<int8_t>(127));
  EXPECT_EQ(out[2], static_cast<int8_t>(128));
  EXPECT_EQ(out[3], static_cast<int8_t>(255));
  EXPECT_EQ(out[4], static_cast<int8_t>(256));
  EXPECT_EQ(out[5], static_cast<int8_t>(1000000));
}

TEST(CastBufferTest, Int8ToUInt16) {
  VecBuffer<int8_t> in({-128, -1, 0, 1, 127});
  auto out = in.cast<uint16_t>();
  ASSERT_EQ(out.size(), in.size());

  EXPECT_EQ(out[0], static_cast<uint16_t>(-128));
  EXPECT_EQ(out[1], static_cast<uint16_t>(-1));
  EXPECT_EQ(out[2], static_cast<uint16_t>(0));
  EXPECT_EQ(out[3], static_cast<uint16_t>(1));
  EXPECT_EQ(out[4], static_cast<uint16_t>(127));
}

TEST(CastBufferTest, FloatSpecialValues) {
  VecBuffer<float> in({std::numeric_limits<float>::infinity(),
                       -std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::quiet_NaN(), -0.0f,
                       std::numeric_limits<float>::min(),
                       std::numeric_limits<float>::max()});

  auto double_out = in.cast<double>();
  ASSERT_EQ(double_out.size(), in.size());
  EXPECT_TRUE(std::isinf(double_out[0]) && double_out[0] > 0);
  EXPECT_TRUE(std::isinf(double_out[1]) && double_out[1] < 0);
  EXPECT_TRUE(std::isnan(double_out[2]));
  EXPECT_DOUBLE_EQ(double_out[3], -0.0);
  EXPECT_DOUBLE_EQ(double_out[4],
                   static_cast<double>(std::numeric_limits<float>::min()));
  EXPECT_DOUBLE_EQ(double_out[5],
                   static_cast<double>(std::numeric_limits<float>::max()));

  auto int_out = in.cast<int32_t>();
  ASSERT_EQ(int_out.size(), in.size());
  // Inf/NaN to int is implementation-defined, so only test ordinary values.
  EXPECT_EQ(int_out[3], static_cast<int32_t>(-0.0f));
  EXPECT_EQ(int_out[4],
            static_cast<int32_t>(std::numeric_limits<float>::min()));
}

// Chained casting - Verify that multiple casts in a row behave as expected.
TEST(CastBufferTest, ChainedCasting) {
  VecBuffer<int32_t> in({-1000, 0, 1000});

  // int32 -> float32 -> uint8 -> int16
  auto float_out = in.cast<float>();
  auto uint8_out = float_out.cast<uint8_t>();
  auto int16_out = uint8_out.cast<int16_t>();

  ASSERT_EQ(int16_out.size(), in.size());
  // The exact result is complex, just verify the chain completes.
  EXPECT_EQ(int16_out.size(), in.size());
}

TEST(CastBufferTest, ExtremeValueCasting) {
  // uint64_max to other types - Test overflow and precision loss.
  VecBuffer<uint64_t> uint64_max({std::numeric_limits<uint64_t>::max()});

  auto uint64_to_int32 = uint64_max.cast<int32_t>();
  EXPECT_EQ(uint64_to_int32[0],
            static_cast<int32_t>(std::numeric_limits<uint64_t>::max()));

  auto uint64_to_float = uint64_max.cast<float>();
  EXPECT_GT(uint64_to_float[0], 1.0e10f);

  // int64_min to other types - Test underflow.
  VecBuffer<int64_t> int64_min({std::numeric_limits<int64_t>::min()});

  auto int64min_to_uint32 = int64_min.cast<uint32_t>();
  EXPECT_EQ(int64min_to_uint32[0],
            static_cast<uint32_t>(std::numeric_limits<int64_t>::min()));

  auto int64min_to_double = int64_min.cast<double>();
  EXPECT_DOUBLE_EQ(int64min_to_double[0],
                   static_cast<double>(std::numeric_limits<int64_t>::min()));
}

TEST(CastBufferTest, EmptyBufferCasting) {
  VecBuffer<float> empty_float;
  auto empty_int = empty_float.cast<int32_t>();
  EXPECT_EQ(empty_int.size(), 0);

  VecBuffer<int16_t> empty_int16(0);
  auto empty_uint8 = empty_int16.cast<uint8_t>();
  EXPECT_EQ(empty_uint8.size(), 0);
}

TEST(CastBufferTest, SameTypeCasting) {
  VecBuffer<int32_t> original({-42, 0, 42});
  auto copy = original.cast<int32_t>();

  ASSERT_EQ(copy.size(), original.size());
  for (size_t i = 0; i < original.size(); ++i) {
    EXPECT_EQ(copy[i], original[i]);
  }
}

TEST(CastBufferTest, LargeBufferCasting) {
  constexpr size_t size = 1000;
  VecBuffer<float> large(size);

  for (size_t i = 0; i < size; ++i) {
    large[i] = static_cast<float>(i) - 500.0f;
  }

  auto large_int = large.cast<int32_t>();
  ASSERT_EQ(large_int.size(), size);

  EXPECT_EQ(large_int[0], -500);
  EXPECT_EQ(large_int[500], 0);
  EXPECT_EQ(large_int[999], 499);
}

// Float precision loss - Casting from double to float should lose precision.
TEST(CastBufferTest, FloatPrecisionLoss) {
  VecBuffer<double> precise({0.1234567890123456789, 1e20, -1e-20});
  auto less_precise = precise.cast<float>();
  auto back_to_double = less_precise.cast<double>();

  ASSERT_EQ(back_to_double.size(), precise.size());

  EXPECT_NE(back_to_double[0], precise[0]);
  EXPECT_NE(back_to_double[1], precise[1]);
  // Very small values might become zero after losing precision.
  EXPECT_NEAR(back_to_double[2], 0.0, 1e-20);
}

// Half-precision conversions - Test round-trip float->half->float conversions.
TEST(CastBufferTest, HalfPrecisionConversions) {
  VecBuffer<float> orig_float({-65504.0f, // Min normal float16
                               65504.0f,  // Max normal float16
                               0.0f, 1.0f, -1.0f, 0.5f, -0.5f,
                               0.1f,    // Will lose precision
                               1.0e-8f, // Will be flushed to zero
                               std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity(),
                               std::numeric_limits<float>::quiet_NaN()});

  auto half_cast = orig_float.cast<half>();
  auto back_to_float = half_cast.cast<float>();

  ASSERT_EQ(back_to_float.size(), orig_float.size());

  EXPECT_NEAR(back_to_float[2], 0.0f, 1e-6f);
  EXPECT_NEAR(back_to_float[3], 1.0f, 1e-6f);
  EXPECT_NEAR(back_to_float[4], -1.0f, 1e-6f);
  EXPECT_NEAR(back_to_float[5], 0.5f, 1e-6f);
  EXPECT_NEAR(back_to_float[6], -0.5f, 1e-6f);

  // Values at float16 limits should be preserved with some tolerance.
  EXPECT_NEAR(back_to_float[0], -65504.0f, 1.0f);
  EXPECT_NEAR(back_to_float[1], 65504.0f, 1.0f);

  EXPECT_NE(back_to_float[7], orig_float[7]);
  EXPECT_NEAR(back_to_float[7], 0.1f, 0.01f);

  EXPECT_NEAR(back_to_float[8], 0.0f, 1e-6f);

  EXPECT_TRUE(std::isinf(back_to_float[9]) && back_to_float[9] > 0);
  EXPECT_TRUE(std::isinf(back_to_float[10]) && back_to_float[10] < 0);
  EXPECT_TRUE(std::isnan(back_to_float[11]));

  // half to int16 - Conversion behavior is implementation-dependent.
  auto half_to_int16 = half_cast.cast<int16_t>();
  EXPECT_EQ(half_to_int16[2], 0);
  EXPECT_EQ(half_to_int16[3], 1);
  EXPECT_EQ(half_to_int16[4], -1);
  EXPECT_EQ(half_to_int16.size(), half_cast.size());
}

// Signed/unsigned conversions - Test how values wrap or truncate.
TEST(CastBufferTest, SignedUnsignedConversions) {
  VecBuffer<int32_t> signed_ints({-100, -1, 0, 1, 100,
                                  std::numeric_limits<int32_t>::min(),
                                  std::numeric_limits<int32_t>::max()});

  auto uint32_cast = signed_ints.cast<uint32_t>();
  ASSERT_EQ(uint32_cast.size(), signed_ints.size());
  EXPECT_EQ(uint32_cast[0], static_cast<uint32_t>(-100));
  EXPECT_EQ(uint32_cast[1], static_cast<uint32_t>(-1)); // Wraps to max uint32
  EXPECT_EQ(uint32_cast[2], 0u);
  EXPECT_EQ(uint32_cast[3], 1u);
  EXPECT_EQ(uint32_cast[4], 100u);
  EXPECT_EQ(uint32_cast[5],
            static_cast<uint32_t>(std::numeric_limits<int32_t>::min()));
  EXPECT_EQ(uint32_cast[6],
            static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));

  // int32 to uint16 - Values are truncated.
  auto uint16_cast = signed_ints.cast<uint16_t>();
  ASSERT_EQ(uint16_cast.size(), signed_ints.size());
  EXPECT_EQ(uint16_cast[0], static_cast<uint16_t>(-100));
  EXPECT_EQ(uint16_cast[1], static_cast<uint16_t>(-1)); // Wraps to 0xFFFF
  EXPECT_EQ(uint16_cast[2], 0u);
  EXPECT_EQ(uint16_cast[3], 1u);
  EXPECT_EQ(uint16_cast[4], 100u);
  EXPECT_EQ(uint16_cast[5],
            static_cast<uint16_t>(std::numeric_limits<int32_t>::min()));
  EXPECT_EQ(uint16_cast[6],
            static_cast<uint16_t>(std::numeric_limits<int32_t>::max()));

  VecBuffer<uint32_t> unsigned_ints({
      0, 1, 100,
      0x7FFFFFFF,                          // Max int32 value
      0x80000000,                          // Just above max int32 value
      std::numeric_limits<uint32_t>::max() // Max uint32
  });

  auto int32_cast = unsigned_ints.cast<int32_t>();
  ASSERT_EQ(int32_cast.size(), unsigned_ints.size());
  EXPECT_EQ(int32_cast[0], 0);
  EXPECT_EQ(int32_cast[1], 1);
  EXPECT_EQ(int32_cast[2], 100);
  EXPECT_EQ(int32_cast[3], static_cast<int32_t>(0x7FFFFFFF));
  EXPECT_EQ(int32_cast[4],
            static_cast<int32_t>(0x80000000)); // Overflows to min int32
  EXPECT_EQ(int32_cast[5],
            static_cast<int32_t>(
                std::numeric_limits<uint32_t>::max())); // Wraps to -1

  auto int16_cast = unsigned_ints.cast<int16_t>();
  ASSERT_EQ(int16_cast.size(), unsigned_ints.size());
  EXPECT_EQ(int16_cast[0], 0);
  EXPECT_EQ(int16_cast[1], 1);
  EXPECT_EQ(int16_cast[2], 100);
  EXPECT_EQ(int16_cast[3], static_cast<int16_t>(0x7FFFFFFF));
  EXPECT_EQ(int16_cast[4], static_cast<int16_t>(0x80000000));
  EXPECT_EQ(int16_cast[5],
            static_cast<int16_t>(std::numeric_limits<uint32_t>::max()));
}

// Overflow/underflow casting - Test behavior when casting values outside the
// representable range.
TEST(CastBufferTest, OverflowUnderflowCasting) {
  VecBuffer<float> big_floats(
      {1.0e10f, -1.0e10f,
       static_cast<float>(std::numeric_limits<int32_t>::max()) * 2.0f,
       static_cast<float>(std::numeric_limits<int32_t>::min()) * 2.0f,
       std::numeric_limits<float>::max(),
       std::numeric_limits<float>::lowest()});

  auto int32_from_big_float = big_floats.cast<int32_t>();
  ASSERT_EQ(int32_from_big_float.size(), big_floats.size());

  // C++ standard: float-to-int overflow is undefined behavior.
  // However, most compilers saturate to the min/max value of the integer type.
  EXPECT_EQ(int32_from_big_float[0], std::numeric_limits<int32_t>::max());
  EXPECT_EQ(int32_from_big_float[1], std::numeric_limits<int32_t>::min());

  VecBuffer<float> neg_floats({-1.0f, -0.5f, -0.0f});
  auto uint32_from_neg = neg_floats.cast<uint32_t>();
  ASSERT_EQ(uint32_from_neg.size(), neg_floats.size());

  EXPECT_EQ(uint32_from_neg[2], 0u); // -0.0f should cast to 0.
}

// Fractional to integer casting - Verify that fractional parts are truncated.
TEST(CastBufferTest, FractionalToIntegerCasting) {
  VecBuffer<double> fractions({0.0, 0.1, 0.49, 0.5, 0.51, 0.99, 1.0, -0.1,
                               -0.49, -0.5, -0.51, -0.99, -1.0});
  auto int_cast = fractions.cast<int32_t>();

  ASSERT_EQ(int_cast.size(), fractions.size());

  EXPECT_EQ(int_cast[0], 0);
  EXPECT_EQ(int_cast[1], 0);
  EXPECT_EQ(int_cast[2], 0);
  EXPECT_EQ(int_cast[3], 0);
  EXPECT_EQ(int_cast[4], 0);
  EXPECT_EQ(int_cast[5], 0);
  EXPECT_EQ(int_cast[6], 1);

  EXPECT_EQ(int_cast[7], 0);
  EXPECT_EQ(int_cast[8], 0);
  EXPECT_EQ(int_cast[9], 0);
  EXPECT_EQ(int_cast[10], 0);
  EXPECT_EQ(int_cast[11], 0);
  EXPECT_EQ(int_cast[12], -1);
}

TEST(CastBufferTest, MixedPrecisionCastChains) {
  VecBuffer<uint8_t> uint8_values(256);
  for (int i = 0; i < 256; i++) {
    uint8_values[i] = static_cast<uint8_t>(i);
  }

  // uint8 -> float32 -> float16 -> float64 -> int32
  auto to_float32 = uint8_values.cast<float>();
  auto to_float16 = to_float32.cast<half>();
  auto to_float64 = to_float16.cast<double>();
  auto to_int32 = to_float64.cast<int32_t>();

  ASSERT_EQ(to_int32.size(), uint8_values.size());

  EXPECT_EQ(to_int32[0], 0);
  EXPECT_EQ(to_int32[1], 1);
  EXPECT_EQ(to_int32[255], 255);

  // int32 -> uint64 -> float32 -> int16
  VecBuffer<int32_t> int32_values({-1000000, -1, 0, 1, 1000000});
  auto to_uint64 = int32_values.cast<uint64_t>();
  auto back_to_float = to_uint64.cast<float>();
  auto to_int16 = back_to_float.cast<int16_t>();

  ASSERT_EQ(to_int16.size(), int32_values.size());

  // Negative values become large positive in uint64, then overflow in
  // float->int16.
  EXPECT_EQ(to_int16[2], 0);
  EXPECT_EQ(to_int16[3], 1);
}

// Subnormal float casting - Test conversion of very small, subnormal float
// values.
TEST(CastBufferTest, SubnormalFloatCasting) {
  float min_normal = std::numeric_limits<float>::min();
  VecBuffer<float> subnormals({
      0.0f,
      std::numeric_limits<float>::denorm_min(), // Smallest positive subnormal
      min_normal / 2,                           // A subnormal value
      min_normal                                // Smallest normal value
  });

  auto to_double = subnormals.cast<double>();
  ASSERT_EQ(to_double.size(), subnormals.size());
  EXPECT_DOUBLE_EQ(to_double[0], 0.0);
  EXPECT_DOUBLE_EQ(to_double[1], static_cast<double>(
                                     std::numeric_limits<float>::denorm_min()));
  EXPECT_DOUBLE_EQ(to_double[2], static_cast<double>(min_normal / 2));
  EXPECT_DOUBLE_EQ(to_double[3], static_cast<double>(min_normal));

  auto to_half = subnormals.cast<half>();
  auto back_to_float = to_half.cast<float>();
  ASSERT_EQ(back_to_float.size(), subnormals.size());
  EXPECT_FLOAT_EQ(back_to_float[0], 0.0f);
}

// Integer rounding - Test rounding of large integers that lose precision in
// float types.
TEST(CastBufferTest, IntegerRoundingInFloatCast) {
  VecBuffer<int64_t> large_ints({
      1000000000000000000LL, // 10^18, will round in float32
      1000000000000000001LL, // 10^18 + 1, will round to same float32
      9007199254740992LL,    // 2^53, max exact double integer
      9007199254740993LL     // 2^53 + 1, will round in double
  });

  auto to_float = large_ints.cast<float>();
  auto back_to_int64 = to_float.cast<int64_t>();

  ASSERT_EQ(back_to_int64.size(), large_ints.size());

  EXPECT_NE(back_to_int64[0], large_ints[0]);
  EXPECT_NE(back_to_int64[1], large_ints[1]);

  auto to_double = large_ints.cast<double>();
  auto double_to_int64 = to_double.cast<int64_t>();

  ASSERT_EQ(double_to_int64.size(), large_ints.size());

  EXPECT_EQ(double_to_int64[2], large_ints[2]);
  EXPECT_NE(double_to_int64[3], large_ints[3]);
}

TEST(CastBufferTest, ComprehensiveTypeConversionGrid) {
  {
    VecBuffer<int8_t> val({42});
    EXPECT_EQ(val.cast<int16_t>()[0], static_cast<int16_t>(42));
    EXPECT_EQ(val.cast<int32_t>()[0], static_cast<int32_t>(42));
    EXPECT_EQ(val.cast<int64_t>()[0], static_cast<int64_t>(42));
    EXPECT_EQ(val.cast<uint8_t>()[0], static_cast<uint8_t>(42));
    EXPECT_EQ(val.cast<uint16_t>()[0], static_cast<uint16_t>(42));
    EXPECT_EQ(val.cast<uint32_t>()[0], static_cast<uint32_t>(42));
    EXPECT_EQ(val.cast<uint64_t>()[0], static_cast<uint64_t>(42));
    EXPECT_NEAR(static_cast<float>(val.cast<half>()[0]), 42.0f, 0.01f);
    EXPECT_FLOAT_EQ(val.cast<float>()[0], 42.0f);
    EXPECT_DOUBLE_EQ(val.cast<double>()[0], 42.0);
  }

  // Negative int8 to unsigned - Test wrap-around behavior.
  {
    VecBuffer<int8_t> neg_val({-42});
    EXPECT_EQ(neg_val.cast<int16_t>()[0], static_cast<int16_t>(-42));
    EXPECT_EQ(neg_val.cast<uint8_t>()[0], static_cast<uint8_t>(-42));
    EXPECT_EQ(neg_val.cast<uint16_t>()[0], static_cast<uint16_t>(-42));
  }

  // uint64_max to other types - Test overflow and precision loss.
  {
    VecBuffer<uint64_t> max_val({std::numeric_limits<uint64_t>::max()});

    EXPECT_EQ(max_val.cast<int8_t>()[0],
              static_cast<int8_t>(std::numeric_limits<uint64_t>::max()));
    EXPECT_EQ(max_val.cast<int16_t>()[0],
              static_cast<int16_t>(std::numeric_limits<uint64_t>::max()));
    EXPECT_EQ(max_val.cast<int32_t>()[0],
              static_cast<int32_t>(std::numeric_limits<uint64_t>::max()));
    EXPECT_EQ(max_val.cast<int64_t>()[0],
              static_cast<int64_t>(std::numeric_limits<uint64_t>::max()));

    EXPECT_GT(max_val.cast<float>()[0], 1.0e19f);
    EXPECT_GT(max_val.cast<double>()[0], 1.0e19);
  }

  // double -> float -> half - Test precision loss chain.
  {
    VecBuffer<double> pi_val({3.14159265358979323846});
    auto pi_float = pi_val.cast<float>();
    auto pi_half = pi_float.cast<half>();

    EXPECT_NE(static_cast<double>(pi_float[0]), pi_val[0]);
    EXPECT_NEAR(pi_float[0], 3.14159265f, 1e-7f);

    auto half_as_float = static_cast<float>(pi_half[0]);
    EXPECT_NE(half_as_float, pi_float[0]);
    EXPECT_NEAR(half_as_float, 3.14f, 0.01f);
  }
}

TEST(BufferTest, AddOperation) {
  py::scoped_interpreter guard{};

  Buffer a({1.0f, 2.0f, 3.0f}, "float32");
  Buffer b({4.0f, 5.0f, 6.0f}, "float32");
  Buffer c = simd_ops::buffer_add(a, b, "float32");
  std::vector<float> expected = {5.0f, 7.0f, 9.0f};
  const auto &c_buf = std::get<VecBuffer<float>>(c.raw());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(c_buf[i], expected[i], 1e-5);
  }
}

TEST(AlignmentTest, FloatAlignment) {
  using T = float;
  const std::size_t alignment = simd_ops::simd_alignment<T>();
  T *ptr = static_cast<T *>(std::aligned_alloc(alignment, alignment * 2));
  ASSERT_NE(ptr, nullptr);

  EXPECT_TRUE(simd_ops::is_aligned<T>(ptr));
  EXPECT_EQ(simd_ops::align_offset<T>(ptr), 0U);

  char *mis_bytes = reinterpret_cast<char *>(ptr) + 1;
  T *mis_ptr = reinterpret_cast<T *>(mis_bytes);
  EXPECT_FALSE(simd_ops::is_aligned<T>(mis_ptr));

  auto addr = reinterpret_cast<std::uintptr_t>(mis_ptr);
  std::size_t expected = alignment - (addr % alignment);
  if (expected == alignment)
    expected = 0;
  expected /= sizeof(T);
  EXPECT_EQ(simd_ops::align_offset<T>(mis_ptr), expected);

  std::free(ptr);
}

TEST(AlignmentTest, Int32Alignment) {
  using T = int32_t;
  const std::size_t alignment = simd_ops::simd_alignment<T>();
  T *ptr = static_cast<T *>(std::aligned_alloc(alignment, alignment * 2));
  ASSERT_NE(ptr, nullptr);

  EXPECT_TRUE(simd_ops::is_aligned<T>(ptr));
  EXPECT_EQ(simd_ops::align_offset<T>(ptr), 0U);

  char *mis_bytes = reinterpret_cast<char *>(ptr) + 1;
  T *mis_ptr = reinterpret_cast<T *>(mis_bytes);
  EXPECT_FALSE(simd_ops::is_aligned<T>(mis_ptr));

  auto addr = reinterpret_cast<std::uintptr_t>(mis_ptr);
  std::size_t expected = alignment - (addr % alignment);
  if (expected == alignment)
    expected = 0;
  expected /= sizeof(T);
  EXPECT_EQ(simd_ops::align_offset<T>(mis_ptr), expected);

  std::free(ptr);
}

TEST(AlignmentTest, SimdWidthAndBytes) {
  EXPECT_EQ(simd_ops::simd_width<float>(), xsimd::batch<float>::size);
  EXPECT_EQ(simd_ops::simd_bytes<float>(),
            xsimd::batch<float>::size * sizeof(float));

  EXPECT_EQ(simd_ops::simd_width<int32_t>(), xsimd::batch<int32_t>::size);
  EXPECT_EQ(simd_ops::simd_bytes<int32_t>(),
            xsimd::batch<int32_t>::size * sizeof(int32_t));
}

TEST(BufferMembersTest, RoundTripAndBounds) {
  py::scoped_interpreter guard{};

  Buffer fbuf(2, "float32");
  fbuf.set_item_typed<float>(0, 1.25f);
  fbuf.set_item_typed<float>(1, -2.5f);
  auto item0 = fbuf.get_item(0);
  auto item1 = fbuf.get_item(1);
  EXPECT_FLOAT_EQ(py::cast<float>(item0), 1.25f);
  EXPECT_FLOAT_EQ(py::cast<float>(item1), -2.5f);

  EXPECT_THROW(static_cast<void>(fbuf.set_item_typed<float>(2, 0.f)),
               std::out_of_range);
  EXPECT_THROW(fbuf.set_item_typed<float>(2, 0.f), std::out_of_range);

  Buffer ibuf(1, "int32");
  ibuf.set_item_typed<int32_t>(0, 42);
  auto item = ibuf.get_item(0);
  EXPECT_EQ(py::cast<int32_t>(item), 42);
}

TEST(BufferMembersTest, ReprContainsInfo) {
  py::scoped_interpreter guard{};
  Buffer buf(3, "int64");
  std::string r = buf.repr();
  EXPECT_NE(r.find("dtype=int64"), std::string::npos);
  EXPECT_NE(r.find("size=3"), std::string::npos);
}

TEST(BufferMembersTest, ArrayInterfaceBasics) {
  py::scoped_interpreter guard{};
  Buffer buf(5, "float32");
  auto ai = buf.array_interface();
  EXPECT_EQ(ai["shape"].cast<py::tuple>()[0].cast<std::size_t>(), 5);
  EXPECT_EQ(ai["typestr"].cast<std::string>(), "<f4");

  Buffer ibuf(2, "int32");
  auto ai2 = ibuf.array_interface();
  EXPECT_EQ(ai2["typestr"].cast<std::string>(), "<i4");
}

TEST(BufferMembersTest, CastBetweenTypes) {
  py::scoped_interpreter guard{};
  Buffer fbuf({1.5f, -2.0f}, "float32");
  Buffer ibuf = fbuf.cast("int32");
  EXPECT_EQ(py::cast<int32_t>(ibuf.get_item(0)), static_cast<int32_t>(1.5f));
  EXPECT_EQ(py::cast<int32_t>(ibuf.get_item(1)), static_cast<int32_t>(-2.0f));

  Buffer i64buf({1, 2, 3}, "int64");
  Buffer f64buf = i64buf.cast("float64");
  EXPECT_DOUBLE_EQ(py::cast<double>(f64buf.get_item(0)), 1.0);
  EXPECT_DOUBLE_EQ(py::cast<double>(f64buf.get_item(1)), 2.0);
  EXPECT_DOUBLE_EQ(py::cast<double>(f64buf.get_item(2)), 3.0);
}

TEST(BufferTest, SetItemOutOfBounds) {
  py::scoped_interpreter guard{};
  Buffer buf(1, "float32");
  EXPECT_THROW(buf.set_item_typed<float>(3, 1.0f), std::out_of_range);
}

TEST(BufferTest, TypeConversions) {
  py::scoped_interpreter guard{};

  Buffer dbuf(1, "float64");
  dbuf.set_item_typed<double>(0, 3.14159);
  EXPECT_NEAR(py::cast<double>(dbuf.get_item(0)), 3.14159, 1e-9);

  Buffer ibuf(1, "int64");
  ibuf.set_item_typed<int64_t>(0, -5);
  EXPECT_EQ(py::cast<int64_t>(ibuf.get_item(0)), -5);

  Buffer fbuf(1, "float32");
  fbuf.set_item_typed<float>(0, 42);
  EXPECT_NEAR(py::cast<float>(fbuf.get_item(0)), 42.0f, 1e-6f);

  fbuf.set_item_typed<float>(0, 5.5f);
  EXPECT_NEAR(py::cast<float>(fbuf.get_item(0)), 5.5f, 1e-6f);
}
