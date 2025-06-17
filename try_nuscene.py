from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-trainval', dataroot="/media/drive-2t/hoangnv83/MV2DFusion/data/nuscenes", verbose=True)

print(nusc.__dict__)

# for my_instance in nusc.instance:
#     instance_token = my_instance['token']
#     print(my_instance.keys())
#     # nusc.render_instance(instance_token)
#     break