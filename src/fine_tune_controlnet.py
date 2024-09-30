from sdxl_architect_module import SDXLArchitectModule
from safetensors.torch import load_file

def main():
    diff_arch_module = SDXLArchitectModule(None).to('cuda')
    img = diff_arch_module.generate("a modern living room with minimalist design")
    img.save('test.png')
    # diff_arch_module.test("a modern living room with minimalist design")

if __name__ == '__main__':
    main()