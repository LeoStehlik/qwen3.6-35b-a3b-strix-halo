#!/usr/bin/env python3
"""Print a persistent systemd user unit. Does not install, enable or start it."""
import serve


def quote(value, execarg=False):
    """Quote a literal systemd argument, escaping expansion and control characters."""
    value = str(value)
    if any(ord(c) < 32 for c in value):
        raise ValueError('Control characters are not allowed in service arguments')
    value = value.replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%')
    if execarg:
        value = value.replace('$', '$$')
    return '"' + value + '"'


def main():
    parser = serve.create_parser()
    parser.description = __doc__
    args = serve.select_default(parser.parse_args())
    args.model_dir = args.model_dir.resolve()
    if args.binary:
        args.binary = args.binary.resolve()
    cmd, _, overrides = serve.command(args)
    print('[Unit]\nDescription=Qwen3.6 ROCmFP4 MTP - promoted Strix Halo prefill\n'
          'After=network-online.target\nWants=network-online.target\n'
          'StartLimitIntervalSec=0\n\n[Service]\nType=simple')
    # Execute the reviewed binary directly; later cookbook edits cannot change it.
    for key, value in sorted(overrides.items()):
        print('Environment=' + quote(key + '=' + value))
    print('UnsetEnvironment=GGML_VK_FA_PROFILE GGML_VK_PERF_LOGGER '
          'GGML_VK_PERF_LOGGER_CONCURRENT GGML_VK_MMID_PROFILE '
          'GGML_VK_FA_F32_OUTPUT GGML_CPU_FA_F32_REFERENCE GGML_VK_FA_MR_TRACE '
          'MESA_VK_TRACE MESA_VK_TRACE_PER_SUBMIT RADV_THREAD_TRACE_CACHE_COUNTERS '
          'RADV_THREAD_TRACE_INSTRUCTION_TIMING LD_PRELOAD LD_LIBRARY_PATH '
          'GGML_VK_PREFER_HOST_MEMORY')
    print('ExecStart=' + ' '.join(quote(a, execarg=True) for a in cmd))
    print('Restart=on-failure\nRestartSec=5\nTimeoutStartSec=900\n'
          'TimeoutStopSec=45\n\n[Install]\nWantedBy=default.target')


if __name__ == '__main__':
    main()
