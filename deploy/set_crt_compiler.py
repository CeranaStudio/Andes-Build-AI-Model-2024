import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--riscv', type=int, default='False')
    parser.add_argument('--toolchain_dir', type=str, required=True)
    args = parser.parse_args()
    riscv = bool(args.riscv)
    if riscv:
        assert os.path.exists('./build/crt/CMakeLists.txt'), 'CMakeLists.txt not found'
        gcc_path = os.path.join(args.toolchain_dir, 'nds64le-linux-glibc-v5d/bin/riscv64-linux-gcc')
        gpp_path = os.path.join(args.toolchain_dir, 'nds64le-linux-glibc-v5d/bin/riscv64-linux-g++')
        assert os.path.exists(gcc_path), gcc_path + ' not found'
        assert os.path.exists(gpp_path), gpp_path + ' not found'
        content = ''
        with open('./build/crt/CMakeLists.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line == 'project(standalone_crt_libs C CXX)\n':
                    content += 'project(standalone_crt_libs C CXX)\n'
                    content += '\n'
                    content += 'set(CMAKE_C_COMPILER ' + gcc_path + ')\n'
                    content += 'set(CMAKE_CXX_COMPILER ' + gpp_path + ')\n'
                else:
                    content += line
        os.remove('./build/crt/CMakeLists.txt')
        with open('./build/crt/CMakeLists.txt', 'w') as f:
            f.write(content)
        print('Set compiler to ' + gcc_path + ' and ' + gpp_path)
        
if __name__ == '__main__':
    main()