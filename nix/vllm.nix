{ lib
, stdenv
, python
, fetchFromGitHub
, fetchpatch
, symlinkJoin
, autoAddDriverRunpath
, # nativeBuildInputs
  which
, # buildInputs
  onednn
, numactl
, llvmPackages
, # internal dependency - for overriding in overlays
  vllm-flash-attn ? null
, cudaSupport
, cudaPackages ? { }
, rocmSupport
, rocmPackages ? { }
, gpuTargets ? [ ]
,
}:

let
  inherit (lib)
    lists
    strings
    trivial
    ;

  inherit (cudaPackages) flags;

  shouldUsePkg =
    pkg: if pkg != null && lib.meta.availableOn stdenv.hostPlatform pkg then pkg else null;

  # see CMakeLists.txt, grepping for CUTLASS_REVISION
  # https://github.com/vllm-project/vllm/blob/v${version}/CMakeLists.txt
  cutlass = fetchFromGitHub {
    name = "cutlass-source";
    owner = "NVIDIA";
    repo = "cutlass";
    tag = "v4.2.1";
    hash = "sha256-iP560D5Vwuj6wX1otJhwbvqe/X4mYVeKTpK533Wr5gY=";
  };

  # FlashMLA's Blackwell (SM100) kernels were developed against CUTLASS v3.9.0
  # (since https://github.com/vllm-project/FlashMLA/commit/9c5dfab6d1746b4a27af14f440e7afd5c01ece68)
  # and are currently incompatible with CUTLASS v4.x APIs. The rest of the vLLM
  # build uses a newer CUTLASS, so we package both versions.
  # See upstream issue: https://github.com/vllm-project/vllm/issues/27425
  # See git submodule commit at:
  # https://github.com/vllm-project/FlashMLA/tree/${flashmla.src.rev}/csrc
  cutlass-flashmla = fetchFromGitHub {
    owner = "NVIDIA";
    repo = "cutlass";
    rev = "147f5673d0c1c3dcf66f78d677fd647e4a020219";
    hash = "sha256-dHQto08IwTDOIuFUp9jwm1MWkFi8v2YJ/UESrLuG71g=";
  };

  flashmla = stdenv.mkDerivation {
    pname = "flashmla";
    # https://github.com/vllm-project/FlashMLA/blob/${src.rev}/setup.py
    version = "1.0.0";

    # grep for GIT_TAG in the following file
    # https://github.com/vllm-project/vllm/blob/v${version}/cmake/external_projects/flashmla.cmake
    src = fetchFromGitHub {
      name = "FlashMLA-source";
      owner = "vllm-project";
      repo = "FlashMLA";
      rev = "c2afa9cb93e674d5a9120a170a6da57b89267208";
      hash = "sha256-pKlwxV6G9iHag/jbu3bAyvYvnu5TbrQwUMFV0AlGC3s=";
    };

    dontConfigure = true;

    # flashmla normally relies on `git submodule update` to fetch cutlass
    buildPhase = ''
      rm -rf csrc/cutlass
      ln -sf ${cutlass-flashmla} csrc/cutlass
    '';

    installPhase = ''
      cp -rva . $out
    '';
  };

  # grep for DEFAULT_TRITON_KERNELS_TAG in the following file
  # https://github.com/vllm-project/vllm/blob/v${version}/cmake/external_projects/triton_kernels.cmake
  triton-kernels = fetchFromGitHub {
    owner = "triton-lang";
    repo = "triton";
    tag = "v3.5.0";
    hash = "sha256-F6T0n37Lbs+B7UHNYzoIQHjNNv3TcMtoXjNrT8ZUlxY=";
  };

  # grep for GIT_TAG in the following file
  # https://github.com/vllm-project/vllm/blob/v${version}/cmake/external_projects/qutlass.cmake
  qutlass = fetchFromGitHub {
    name = "qutlass-source";
    owner = "IST-DASLab";
    repo = "qutlass";
    rev = "830d2c4537c7396e14a02a46fbddd18b5d107c65";
    hash = "sha256-aG4qd0vlwP+8gudfvHwhtXCFmBOJKQQTvcwahpEqC84=";
  };

  vllm-flash-attn' = lib.defaultTo
    (stdenv.mkDerivation {
      pname = "vllm-flash-attn";
      # https://github.com/vllm-project/flash-attention/blob/${src.rev}/vllm_flash_attn/__init__.py
      version = "2.7.2.post1";

      # grep for GIT_TAG in the following file
      # https://github.com/vllm-project/vllm/blob/v${version}/cmake/external_projects/vllm_flash_attn.cmake
      src = fetchFromGitHub {
        name = "flash-attention-source";
        owner = "vllm-project";
        repo = "flash-attention";
        rev = "188be16520ceefdc625fdf71365585d2ee348fe2";
        hash = "sha256-Osec+/IF3+UDtbIhDMBXzUeWJ7hDJNb5FpaVaziPSgM=";
      };

      patches = [
        # fix Hopper build failure
        # https://github.com/Dao-AILab/flash-attention/pull/1719
        # https://github.com/Dao-AILab/flash-attention/pull/1723
        (fetchpatch {
          url = "https://github.com/Dao-AILab/flash-attention/commit/dad67c88d4b6122c69d0bed1cebded0cded71cea.patch";
          hash = "sha256-JSgXWItOp5KRpFbTQj/cZk+Tqez+4mEz5kmH5EUeQN4=";
        })
        (fetchpatch {
          url = "https://github.com/Dao-AILab/flash-attention/commit/e26dd28e487117ee3e6bc4908682f41f31e6f83a.patch";
          hash = "sha256-NkCEowXSi+tiWu74Qt+VPKKavx0H9JeteovSJKToK9A=";
        })
      ];

      dontConfigure = true;

      # vllm-flash-attn normally relies on `git submodule update` to fetch cutlass and composable_kernel
      buildPhase = ''
        rm -rf csrc/cutlass
        ln -sf ${cutlass} csrc/cutlass
      ''
      + lib.optionalString rocmSupport ''
        rm -rf csrc/composable_kernel;
        ln -sf ${rocmPackages.composable_kernel} csrc/composable_kernel
      '';

      installPhase = ''
        cp -rva . $out
      '';
    })
    vllm-flash-attn;

  cpuSupport = !cudaSupport && !rocmSupport;

  # https://github.com/pytorch/pytorch/blob/v2.9.1/torch/utils/cpp_extension.py#L2407-L2410
  supportedTorchCudaCapabilities =
    let
      real = [
        "3.5"
        "3.7"
        "5.0"
        "5.2"
        "5.3"
        "6.0"
        "6.1"
        "6.2"
        "7.0"
        "7.2"
        "7.5"
        "8.0"
        "8.6"
        "8.7"
        "8.9"
        "9.0"
        "9.0a"
        "10.0"
        "10.0a"
        "10.3"
        "10.3a"
        "11.0"
        "11.0a"
        "12.0"
        "12.0a"
        "12.1"
        "12.1a"
      ];
      ptx = lists.map (x: "${x}+PTX") real;
    in
    real ++ ptx;

  # NOTE: The lists.subtractLists function is perhaps a bit unintuitive. It subtracts the elements
  #   of the first list *from* the second list. That means:
  #   lists.subtractLists a b = b - a

  # For CUDA
  supportedCudaCapabilities = lists.intersectLists flags.cudaCapabilities supportedTorchCudaCapabilities;
  unsupportedCudaCapabilities = lists.subtractLists supportedCudaCapabilities flags.cudaCapabilities;

  isCudaJetson = cudaSupport && cudaPackages.flags.isJetsonBuild;

  # Use trivial.warnIf to print a warning if any unsupported GPU targets are specified.
  gpuArchWarner =
    supported: unsupported:
    trivial.throwIf (supported == [ ])
      (
        "No supported GPU targets specified. Requested GPU targets: "
        + strings.concatStringsSep ", " unsupported
      )
      supported;

  # Create the gpuTargetString.
  gpuTargetString = strings.concatStringsSep ";" (
    if gpuTargets != [ ] then
    # If gpuTargets is specified, it always takes priority.
      gpuTargets
    else if cudaSupport then
      gpuArchWarner supportedCudaCapabilities unsupportedCudaCapabilities
    else if rocmSupport then
      rocmPackages.clr.localGpuTargets or rocmPackages.clr.gpuTargets
    else
      throw "No GPU targets specified"
  );

  mergedCudaLibraries = with cudaPackages; [
    cuda_cudart # cuda_runtime.h, -lcudart
    cuda_cccl
    libcurand # curand_kernel.h
    libcusparse # cusparse.h
    libcusolver # cusolverDn.h
    cuda_nvtx
    cuda_nvrtc
    # cusparselt # cusparseLt.h
    libcublas
  ];

  # header path ends up missing rocthrust & its deps
  rocmExtraIncludeFlags = lib.concatMapStringsSep " " (pkg: "-I${lib.getInclude pkg}/include") [
    rocmPackages.rocthrust
    rocmPackages.rocprim
    rocmPackages.hipcub
  ];

  # Some packages are not available on all platforms
  nccl = shouldUsePkg (cudaPackages.nccl or null);

  getAllOutputs = p: [
    (lib.getBin p)
    (lib.getLib p)
    (lib.getDev p)
  ];

in
final: prev: {
  vllm = (prev.vllm.override { stdenv = final.torch.stdenv; }).overrideAttrs (old: {
    patches = [
      ./patches/vllm-no-openmp.patch
      ./patches/vllm-preserve-pythonpath.patch
      ./patches/vllm-rocm-reqs.patch
      ./patches/vllm-setup-py-env.patch
    ];

    postPatch = ''
      # Remove vendored pynvml entirely
      rm vllm/third_party/pynvml.py
      substituteInPlace tests/utils.py \
        --replace-fail \
          "from vllm.third_party.pynvml import" \
          "from pynvml import"
      substituteInPlace vllm/utils/import_utils.py \
        --replace-fail \
          "import vllm.third_party.pynvml as pynvml" \
          "import pynvml"

      # pythonRelaxDeps does not cover build-system
      substituteInPlace pyproject.toml \
        --replace-fail "torch ==" "torch >=" \
        --replace-fail "setuptools>=77.0.3,<81.0.0" "setuptools" \
        --replace-fail "grpcio-tools==1.78.0" "grpcio"

      # Ignore the python version check because it hard-codes minor versions and
      # lags behind `ray`'s python interpreter support
      substituteInPlace CMakeLists.txt \
        --replace-fail \
          'set(PYTHON_SUPPORTED_VERSIONS' \
          'set(PYTHON_SUPPORTED_VERSIONS "${lib.versions.majorMinor python.version}"'
    '';

    nativeBuildInputs = old.nativeBuildInputs ++ [
      which
    ]
      ++ lib.optionals rocmSupport [
      rocmPackages.hipcc
    ]
      ++ lib.optionals cudaSupport [
      cudaPackages.cuda_nvcc
      autoAddDriverRunpath
    ]
      ++ lib.optionals isCudaJetson [
      cudaPackages.autoAddCudaCompatRunpath
    ];

    buildInputs =
      lib.optionals cpuSupport [
        onednn
      ]
      ++ lib.optionals (cpuSupport && stdenv.hostPlatform.isLinux) [
        numactl
      ]
      ++ lib.optionals cudaSupport (
        mergedCudaLibraries
        ++ (with cudaPackages; [
          nccl
          cudnn
          libcufile
        ])
      )
      ++ lib.optionals rocmSupport (
        with rocmPackages;
        [
          clr
          rocthrust
          rocprim
          hipsparse
          hipblas
          rocrand
          hiprand
          rocblas
          miopen-hip
          hipfft
          hipcub
          hipsolver
          rocsolver
          hipblaslt
          rocm-runtime
        ]
      )
      ++ lib.optionals stdenv.cc.isClang [
        llvmPackages.openmp
      ];


    dontUseCmakeConfigure = true;
    cmakeFlags = [
    ]
    ++ lib.optionals cudaSupport [
      (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_CUTLASS" "${lib.getDev cutlass}")
      (lib.cmakeFeature "FLASH_MLA_SRC_DIR" "${lib.getDev flashmla}")
      (lib.cmakeFeature "VLLM_FLASH_ATTN_SRC_DIR" "${lib.getDev vllm-flash-attn'}")
      (lib.cmakeFeature "QUTLASS_SRC_DIR" "${lib.getDev qutlass}")
      (lib.cmakeFeature "TORCH_CUDA_ARCH_LIST" "${gpuTargetString}")
      (lib.cmakeFeature "CUTLASS_NVCC_ARCHS_ENABLED" "${cudaPackages.flags.cmakeCudaArchitecturesString}")
      (lib.cmakeFeature "CUDA_TOOLKIT_ROOT_DIR" "${symlinkJoin {
      name = "cuda-merged-${cudaPackages.cudaMajorMinorVersion}";
      paths = builtins.concatMap getAllOutputs mergedCudaLibraries;
    }}")
      (lib.cmakeFeature "CAFFE2_USE_CUDNN" "ON")
      (lib.cmakeFeature "CAFFE2_USE_CUFILE" "ON")
      (lib.cmakeFeature "CUTLASS_ENABLE_CUBLAS" "ON")
    ];

    env =
      lib.optionalAttrs cudaSupport
        {
          VLLM_TARGET_DEVICE = "cuda";
          CUDA_HOME = "${lib.getDev cudaPackages.cuda_nvcc}";
          TRITON_KERNELS_SRC_DIR = "${lib.getDev triton-kernels}/python/triton_kernels/triton_kernels";
        }
      // lib.optionalAttrs rocmSupport {
        VLLM_TARGET_DEVICE = "rocm";
        PYTORCH_ROCM_ARCH = gpuTargetString;
        # vLLM's CMake logic checks `ROCM_PATH` to decide whether HIP/ROCm is available.
        ROCM_PATH = "${rocmPackages.clr}";
        TRITON_KERNELS_SRC_DIR = "${lib.getDev triton-kernels}/python/triton_kernels/triton_kernels";
        HIPFLAGS = rocmExtraIncludeFlags;
        CXXFLAGS = rocmExtraIncludeFlags;
      }
      // lib.optionalAttrs cpuSupport {
        VLLM_TARGET_DEVICE = "cpu";
        FETCHCONTENT_SOURCE_DIR_ONEDNN = "${onednn.src}";
      };

    preConfigure = ''
      # See: https://github.com/vllm-project/vllm/blob/v0.7.1/setup.py#L75-L109
      # There's also NVCC_THREADS but Nix/Nixpkgs doesn't really have this concept.
      export MAX_JOBS="$NIX_BUILD_CORES"
    '';

    pythonRelaxDeps = true;

    pythonImportsCheck = [ "vllm" ];

    passthru = {
      # make internal dependency available to overlays
      vllm-flash-attn = vllm-flash-attn';
      # updates the cutlass fetcher instead
      skipBulkUpdate = true;
    };
  });
}
