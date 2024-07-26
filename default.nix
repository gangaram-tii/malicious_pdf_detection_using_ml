{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.gcc
    pkgs.python311Full
    pkgs.python311Packages.virtualenv
    pkgs.python311Packages.numpy
    pkgs.python311Packages.pandas
    pkgs.python311Packages.matplotlib
    pkgs.python311Packages.pytorch
    pkgs.python311Packages.pip
    pkgs.python311Packages.xgboost
  ];

  shellHook = ''
    if [ ! -d .venv ]; then
      virtualenv .venv
      #export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=1
      source .venv/bin/activate
      pip install scikit-learn
      pip install textblob pdfid
    else
      source .venv/bin/activate
      #exportSKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=1
      pip install scikit-learn
      pip install textblob pdfid
    fi
    echo "Welcome to your Python development environment."
  '';
}

