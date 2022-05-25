import os, string, shutil,re
import pefile

#pefile_path = r'./api-ms-win-core-string-l1-1-0.dll'

# 普通文件不能读取
#pefile_path = r'./data.txt'
#pefile_path = r'./a.out'
pefile_path = r'./test.exe'

pefile_info = pefile.PE(pefile_path)
print(pefile_path)
#print(pefile_info)

# 保存完整信息到文件（窗口显示不全）
#fd = open(r'./pefile_info.txt', 'w') 
#fd.write(str(pefile_info)) 
#fd.close()

print(pefile_info.OPTIONAL_HEADER.AddressOfEntryPoint)

# 节表
for section in pefile_info.sections:
    print(section)

print('-' * 100)
# 获取第三节点的Characteristics信息
print(hex(pefile_info.sections[2].Characteristics))

print('-' * 100)
# 导入表
for importeddll in pefile_info.DIRECTORY_ENTRY_IMPORT:
    print(importeddll.dll)
    ##or use
    #print(pe.DIRECTORY_ENTRY_IMPORT[0].dll)
    for importedapi in importeddll.imports:
        print(importedapi.name)
    ##or use
    #print(pe.DIRECTORY_ENTRY_IMPORT[0].imports[0].name)

print('-' * 100)
# 节表
for section in pefile_info.sections:
    print(section.Name, hex(section.VirtualAddress), hex(section.Misc_VirtualSize), section.SizeOfRawData)









