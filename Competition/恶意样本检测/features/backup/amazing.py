# 太奇怪
import pefile

# 加了fast_load=True居然就无法获取了
pe = pefile.PE('./AIFirst_data\\train\\black\\Backdoor.Win32.Agent.bjkd_4713.exe')
#pe = pefile.PE('./AIFirst_data\\train\\black\\Backdoor.Win32.Agent.bjkd_4713.exe', fast_load=True)
#print(pe)

print(pe.OPTIONAL_HEADER.DATA_DIRECTORY)
print('-'*50)
print(pe.OPTIONAL_HEADER.DATA_DIRECTORY[1].Size)
print('-'*50)

NumberOfFunctions = 0
print(pe.DIRECTORY_ENTRY_IMPORT)
for entry in pe.DIRECTORY_ENTRY_IMPORT:
    NumberOfFunctions += len(entry.imports)
print(NumberOfFunctions)