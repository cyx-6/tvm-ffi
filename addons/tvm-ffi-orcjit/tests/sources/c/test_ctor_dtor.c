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

/*
 * Pure C version of the constructor/destructor test.
 * Uses TVMFFIFunctionGetGlobal + TVMFFIFunctionCall to call the host
 * "append_log" function from constructors and destructors.
 */
#include <tvm/ffi/c_api.h>

static void puts_log(const char* msg) {
  TVMFFIByteArray func_name;
  TVMFFIObjectHandle func_handle = NULL;
  TVMFFIAny call_args[1];
  TVMFFIAny call_result;

  func_name.data = "append_log";
  func_name.size = 10;
  if (TVMFFIFunctionGetGlobal(&func_name, &func_handle) != 0) return;

  call_args[0].type_index = kTVMFFIRawStr;
  call_args[0].zero_padding = 0;
  call_args[0].v_c_str = msg;

  call_result.type_index = kTVMFFINone;
  call_result.zero_padding = 0;
  call_result.v_int64 = 0;
  TVMFFIFunctionCall(func_handle, call_args, 1, &call_result);
  TVMFFIObjectDecRef((TVMFFIObject*)func_handle);
}

typedef void (*ctor_t)(void);
typedef void (*dtor_t)(void);

/* init_array / fini_array — works on all platforms */
__attribute__((constructor)) void init_array(void) { puts_log("<init_array>"); }
__attribute__((constructor(101))) void init_array_101(void) { puts_log("<init_array.101>"); }
__attribute__((constructor(102))) void init_array_102(void) { puts_log("<init_array.102>"); }
__attribute__((constructor(103))) void init_array_103(void) { puts_log("<init_array.103>"); }

__attribute__((destructor)) void fini_array(void) { puts_log("<fini_array>"); }
__attribute__((destructor(101))) void fini_array_101(void) { puts_log("<fini_array.101>"); }
__attribute__((destructor(102))) void fini_array_102(void) { puts_log("<fini_array.102>"); }
__attribute__((destructor(103))) void fini_array_103(void) { puts_log("<fini_array.103>"); }

/* ELF-specific: explicit .ctors/.dtors section placements with priorities */
#ifdef __ELF__
static void ctors(void) { puts_log("<ctors>"); }
__attribute__((section(".ctors"), used)) static ctor_t ctors_ptr = ctors;

static void ctors_101(void) { puts_log("<ctors.101>"); }
__attribute__((section(".ctors.101"), used)) static ctor_t ctors_1_ptr = ctors_101;

static void ctors_102(void) { puts_log("<ctors.102>"); }
__attribute__((section(".ctors.102"), used)) static ctor_t ctors_2_ptr = ctors_102;

static void ctors_103(void) { puts_log("<ctors.103>"); }
__attribute__((section(".ctors.103"), used)) static ctor_t ctors_3_ptr = ctors_103;

static void dtors(void) { puts_log("<dtors>"); }
__attribute__((section(".dtors"), used)) static dtor_t dtors_ptr = dtors;

static void dtors_101(void) { puts_log("<dtors.101>"); }
__attribute__((section(".dtors.101"), used)) static dtor_t dtors_1_ptr = dtors_101;

static void dtors_102(void) { puts_log("<dtors.102>"); }
__attribute__((section(".dtors.102"), used)) static dtor_t dtors_2_ptr = dtors_102;

static void dtors_103(void) { puts_log("<dtors.103>"); }
__attribute__((section(".dtors.103"), used)) static dtor_t dtors_3_ptr = dtors_103;
#endif /* __ELF__ */

/* main: callable entry point */
TVM_FFI_DLL_EXPORT int __tvm_ffi_main(
    void* self, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result) {
  puts_log("<main>");
  result->type_index = kTVMFFINone;
  result->zero_padding = 0;
  result->v_int64 = 0;
  return 0;
}
