{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    python312
    python312Packages.pip
    python312Packages.virtualenv
  ];

  buildInputs = with pkgs; [
    stdenv.cc.cc.lib
    zlib
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
    ]}:/run/opengl-driver/lib:$LD_LIBRARY_PATH"

    if [ ! -d .venv ]; then
      echo "Creating virtualenv..."
      python -m venv .venv
    fi
    source .venv/bin/activate
  '';
}
