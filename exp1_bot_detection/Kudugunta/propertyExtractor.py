import os
import numpy as np
import torch

id = []
statuses_count = []
followers_count = []
friends_count = []
favourites_count = []
listed_count = []
default_profile = []
geo_enabled = []
profile_uses_background_image = []
verified = []
protected = []

for i in range(1, 41):
    files = os.listdir('./profile/' + str(i))
    for file in files:
        f = open('./profile/' + str(i) + '/' + file, 'r', encoding='UTF-8')
        wnn = f.readlines()
        # print('this is a error ID with error code' in wnn[0])
        if 'this is a error ID with error code' in wnn[0]:
            # print('Error!' + file)
            f.close()
            continue
        # print('#' + str(len(id)))
        id.append(file[: -8])
        # print(len(id))
        flag = 0
        for line in wnn:
            if line == 'statuses_count\n' and len(statuses_count) != len(id):
                flag = 1
                continue
            if flag == 1:
                statuses_count.append(float(line))
                flag = 0
                continue
            if line == 'followers_count\n' and len(followers_count) != len(id):
                flag = 2
                continue
            if flag == 2:
                followers_count.append(float(line))
                flag = 0
                continue
            if line == 'friends_count\n' and len(friends_count) != len(id):
                flag = 3
                continue
            if flag == 3:
                friends_count.append(float(line))
                flag = 0
                continue
            if line == 'favourites_count\n' and len(favourites_count) != len(id):
                flag = 4
                continue
            if flag == 4:
                favourites_count.append(float(line))
                flag = 0
                continue
            if line == 'listed_count\n' and len(listed_count) != len(id):
                flag = 5
                continue
            if flag == 5:
                listed_count.append(float(line))
                flag = 0
                continue
            if line == 'default_profile\n' and len(default_profile) != len(id):
                flag = 6
                continue
            if flag == 6:
                default_profile.append(1.) if line == 'True\n' else default_profile.append(0.)
                flag = 0
                continue
            if line == 'geo_enabled\n' and len(geo_enabled) != len(id):
                flag = 7
                continue
            if flag == 7:
                geo_enabled.append(0.) if line == 'False\n' else geo_enabled.append(1.)
                flag = 0
                continue
            if line == 'profile_background_image_url\n' and len(profile_uses_background_image) != len(id):
                flag = 8
                continue
            if flag == 8:
                profile_uses_background_image.append(0.) if line == 'None\n' else profile_uses_background_image.append(1.)
                flag = 0
                continue
            if line == 'verified\n' and len(verified) != len(id):
                flag = 9
                continue
            if flag == 9:
                verified.append(0.) if line == 'False\n' else verified.append(1.)
                flag = 0
                continue
            if line == 'protected\n' and len(protected) != len(id):
                flag = 10
                continue
            if flag == 10:
                protected.append(0.) if line == 'False\n' else protected.append(1.)
                flag = 0
                continue
        f.close()
        if len(statuses_count) == len(followers_count)\
                == len(friends_count) == len(favourites_count)\
                == len(listed_count) == len(default_profile)\
                == len(geo_enabled) == len(profile_uses_background_image)\
                == len(verified) == len(protected) == len(id):
            pass
        else:
            print(str(i) + '#' + str(file))
            print(len(statuses_count))
            print(len(followers_count))
            print(len(friends_count))
            print(len(favourites_count))
            print(len(listed_count))
            print(len(default_profile))
            print(len(geo_enabled))
            print(len(profile_uses_background_image))
            print(len(verified))
            print(len(protected))
            print(len(id))
            assert 1 == 0
statuses_count = (statuses_count - np.mean(statuses_count)) / np.sqrt(np.var(statuses_count))
followers_count = (followers_count - np.mean(followers_count)) / np.sqrt(np.var(followers_count))
friends_count = (friends_count - np.mean(friends_count)) / np.sqrt(np.var(friends_count))
favourites_count = (favourites_count - np.mean(favourites_count)) / np.sqrt(np.var(favourites_count))
listed_count = (listed_count - np.mean(listed_count)) / np.sqrt(np.var(listed_count))

assert len(statuses_count) == len(followers_count)\
    == len(friends_count) == len(favourites_count)\
    == len(listed_count) == len(default_profile)\
    == len(geo_enabled) == len(profile_uses_background_image)\
    == len(verified) == len(protected) == len(id)

for i in range(len(id)):
    property = torch.tensor([[statuses_count[i],  # statuses_count
                              followers_count[i],  # followers_count
                              friends_count[i],  # friends_count
                              favourites_count[i],  # favourites_count
                              listed_count[i],  # listed_count
                              default_profile[i],  # default_profile
                              geo_enabled[i],  # geo_enabled
                              profile_uses_background_image[i],  # profile_use_background_image
                              protected[i],  # protected
                              verified[i]  # verified
                              ]])
    torch.save(property, './property/' + str(id[i]) + '.pth')
