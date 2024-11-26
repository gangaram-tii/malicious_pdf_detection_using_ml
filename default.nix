{ pkgs ? import <nixpkgs> {} }:

pkgs.python3.pkgs.buildPythonPackage rec {
  pname = "pdfscan";
  version = "0.0.1";

  src = ./.;

  propagatedBuildInputs = [
    pkgs.python3Packages.pandas
    pkgs.python3Packages.xgboost
  ];

  meta = with pkgs.lib; {
    description = "A Python package to generate secure configuration for systemd service.";
    license = licenses.mit;
    maintainers = with maintainers; [ ];
  };
}
