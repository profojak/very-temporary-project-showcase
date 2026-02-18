{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = inputs: let
    system = "aarch64-darwin";
    pkgs = import inputs.nixpkgs { inherit system; };
  in {
    devShells.${system}.default = pkgs.mkShell.override
      { stdenv = pkgs.gcc14Stdenv; }
      {
        packages = with pkgs; [ boost ];
      };
  };
}
