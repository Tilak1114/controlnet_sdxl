from diffusion_architect_module import DiffusionArchitectModule
from safetensors.torch import load_file

def main():

    diff_arch_module = DiffusionArchitectModule().to('cuda')
    diff_arch_module.test()

if __name__ == '__main__':
    main()