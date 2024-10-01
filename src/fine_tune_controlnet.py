from sdxl_module import SDXLModule
from safetensors.torch import load_file

def main():
    diff_arch_module = SDXLModule(None).to('cuda')
    img = diff_arch_module.generate("a modern living room with minimalist design")
    img.save('test.png')
    # diff_arch_module.test("a modern living room with minimalist design")

if __name__ == '__main__':
    main()