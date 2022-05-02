import os
import numpy as np
import torch
from PIL import Image

from config import DB, DATA_DIR

## This function creates a balanced set of patches that match and set of nonmatches
# These pair of images are later read by dataloader to feed into the network
def matchMaker(dataDir):
    
    print("Match making!")
    imgsAdrsList = list(sorted(os.listdir(os.path.join(dataDir, DB))))
    labels = np.genfromtxt(os.path.join(dataDir, (DB+"_labels.csv")), delimiter=',')

    # split = round(len(imgsAdrsList)/2)
    # CREATING MATCHES
    matches = np.array([[0,0]], dtype='U6')
    i=0
    j = 0
    totalProcess = ((len(imgsAdrsList)-j)/2) - len(matches)
    while len(matches) < ((len(imgsAdrsList)-j)/2): # condition for balancing matches and nonmatches
        if i % 1000 == 0:
            print("\r", end="")
            # print("Matchmaker: ", str(len(matches)), ' matches',  end="")
            # print("Matchmaker: ", str(int((totalProcess-(((len(imgsAdrsList)-j)/2) - len(matches)))/totalProcess)), '% ',  end="")
            print("Matchmaker: matches ", str(len(matches)), ', left: ', str((len(imgsAdrsList)-j)/2),  end="")

        if labels[i] != labels[j]:
            i = j
            continue
        
        while labels[j] == labels[i]:
            j = j+1

            if labels[j] == labels[i]:
                if j-i == 1:
                    matches = np.concatenate((matches, [[imgsAdrsList[i], imgsAdrsList[j]]]), axis=0)

                if j-i == 3:
                    i = i+2
                    matches = np.concatenate((matches, [[imgsAdrsList[i], imgsAdrsList[j]]]), axis=0)

            else:
                break
    split = j
    matches = np.delete(matches, 0, 0) #remove the first fake row

    #CREATING NON-MATCHES
    lenLeft = (len(imgsAdrsList)-split)
    tmp_nonmatches = np.concatenate(([imgsAdrsList[(split+round(lenLeft/2)):len(imgsAdrsList)]], [imgsAdrsList[split:(split+round(lenLeft/2))]]), axis=0)
    tmp_nonmatches = tmp_nonmatches.T
    nonmatches = np.zeros(tmp_nonmatches.size, dtype='U6')
    nonmatches = tmp_nonmatches

    print("Created " + str(len(matches)) + " matches and " + str(len(nonmatches)) + " non-matches")

    return matches, nonmatches


def DBcombiner(DBs):
    AllMatches = np.array([[0,0]], dtype='U6')
    AllNonMatches = np.array([[0,0]], dtype='U6')
    for db in DBs:
        #making patch matches
        if os.path.exists(DATA_DIR+db+"_matches.csv"):  #read from saved files if already available
            print("skipping matchmaking process")
            matchSet = np.genfromtxt(DATA_DIR+db+"_matches.csv", delimiter=',')
            nonmatchSet = np.genfromtxt(DATA_DIR+db+"_nonmatches.csv", delimiter=',')
        else:
            matchSet, nonmatchSet = matchMaker(DATA_DIR)    # o.w. compute
            np.savetxt(DATA_DIR+db+"_matches.csv", matchSet, delimiter=",", fmt='%s')
            np.savetxt(DATA_DIR+db+"_nonmatches.csv", nonmatchSet, delimiter=",", fmt='%s')
        
        matches_path = os.path.join(DATA_DIR, (DB+"_matches.csv"))
        nonmatches_path = os.path.join(DATA_DIR, (DB+"_nonmatches.csv"))
        matches = np.genfromtxt(matches_path, delimiter=',', dtype=None, encoding=None)
        nonmatches = np.genfromtxt(nonmatches_path, delimiter=',', dtype=None, encoding=None)

        matches = np.core.defchararray.add(DATA_DIR+db+'/', matches)
        nonmatches = np.core.defchararray.add(DATA_DIR+db+'/', nonmatches)

        matches = np.delete(AllMatches, 0, 0) #remove the first fake row
        matches = np.delete(AllNonMatches, 0, 0) #remove the first fake row

        AllMatches = np.concatenate(AllMatches, matches, axis=0)
        AllNonMatches = np.concatenate(AllNonMatches, nonmatches, axis=0)

    

    np.savetxt(DATA_DIR+"All_matches.csv", matchSet, delimiter=",", fmt='%s')
    np.savetxt(DATA_DIR+"All_nonmatches.csv", nonmatchSet, delimiter=",", fmt='%s')


class patchSet(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        # self.imgsAdrsList = list(sorted(os.listdir(os.path.join(root, DB))))
        # self.labelsAdrs = os.path.join(root, (DB+"_labels.csv"))
        matches_path = os.path.join(root, (DB+"_matches.csv"))
        nonmatches_path = os.path.join(root, (DB+"_nonmatches.csv"))
        matches = np.genfromtxt(matches_path, delimiter=',', dtype=None, encoding=None)
        nonmatches = np.genfromtxt(nonmatches_path, delimiter=',', dtype=None, encoding=None)

        self.patchPairs = np.concatenate((matches, nonmatches),axis=0)
        self.labels = np.concatenate((np.ones(len(matches)), np.zeros(len(nonmatches))))


    def __getitem__(self, idx):
        
        # load images
        img_path1 = os.path.join(self.root, DB, self.patchPairs[idx,0])
        img1 = Image.open(img_path1).convert('L')
        
        img_path2 = os.path.join(self.root, DB, self.patchPairs[idx,1])
        img2 = Image.open(img_path2).convert('L')

        #pre-process (intensity normalization)
        img1 = (img1.astype(np.float32) - 128) / 160
        img2 = (img2.astype(np.float32) - 128) / 160

        img = np.concatenate((img1, img2), axis=1)

        # img = torch.as_tensor(imgs, dtype=torch.int32)
        target = self.labels[idx]
        target = torch.as_tensor(target, dtype=torch.long)

        img = self.transforms(img)
        
        return img, target

    def __len__(self):
        return len(self.patchPairs)

