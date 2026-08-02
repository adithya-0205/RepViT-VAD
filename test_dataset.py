from video_dataset import VideoDataset

train = VideoDataset(train=True)
val = VideoDataset(train=False)

print("Train videos :", len(train))
print("Validation videos :", len(val))

x,y = train[0]

print(x.shape)
print(y)