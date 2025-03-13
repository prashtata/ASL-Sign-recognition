import os
import shutil

dirs = os.listdir("/home/prashtata/gradschool/asl/dataset/MP_data")
# # print(dirs)
# target = []
# for dir in dirs:
#     if dir[-4:]!="_aug":
#             target.append(dir)

# for dir in target:
#     source_names = os.listdir(f"/home/prashtata/gradschool/asl/dataset/MP_data/{dir}_aug")
#     for name in source_names:
#         if os.path.exists(f"/home/prashtata/gradschool/asl/dataset/MP_data/{dir}/{name}_aug"): continue
#         else:
#             os.mkdir(f"/home/prashtata/gradschool/asl/dataset/MP_data/{dir}/{name}_aug")
#             file = os.listdir(f"/home/prashtata/gradschool/asl/dataset/MP_data/{dir}_aug/{name}")
#             if len(file) == 0: continue
#             else: shutil.move(f"/home/prashtata/gradschool/asl/dataset/MP_data/{dir}_aug/{name}/{file[0]}", f"/home/prashtata/gradschool/asl/dataset/MP_data/{dir}/{name}_aug")

for dir in dirs:
    if dir[-4:]=="_aug":
            shutil.rmtree(f"/home/prashtata/gradschool/asl/dataset/MP_data/{dir}")
