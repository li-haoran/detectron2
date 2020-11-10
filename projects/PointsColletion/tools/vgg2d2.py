import torch
import sys
import pickle
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def convert(path1,path2,depth='D',bn=True):

    obj = torch.load(path1, map_location="cpu")
    conv_dict={}
    bn_dict={}
    k=0
    s=1
    sk=0
    for i,x in enumerate(cfgs[depth]):
        if x=='M':
            s+=1
            sk=0
            k+=1
        else:
            conv_dict[k]=(s,sk)
            k+=1
            if bn:
                bn_dict[k]=(s,sk)
                k+=1
            sk+=1
            k+=1
    print(conv_dict,bn_dict)
    newmodel = {}

    for k in list(obj.keys()):
        old_k = k       
        if 'features' in k:
            id=int(k.split('.')[1])
            if id in conv_dict:
                stage_id,tmp=conv_dict[id]
                k = k.replace("features.{}".format(id), "vgg_block{}.{}".format(stage_id,tmp))
            elif id in bn_dict:
                stage_id,tmp=bn_dict[id]
                k=k.replace("features.{}".format(id), "vgg_block{}.{}.bn".format(stage_id,tmp))

        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}

    with open(path2, "wb") as f:
        pickle.dump(res, f)
    if obj:
        print("Unconverted keys:", obj.keys())   

if __name__=='__main__':
    convert(sys.argv[1],sys.argv[2]) 


