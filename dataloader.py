import os
import numpy as np
import torch
from PIL import Image

from config import DBs, DATA_DIR, TEMPS_DIR

## This function creates a balanced set of patches that match and set of nonmatches
# These pair of images are later read by dataloader to feed into the network
def matchMaker(dataDir, DB):
    
    print("Match making for " + DB + " dataset")
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
    # if ((len(imgsAdrsList)-(split+round(lenLeft/2))) != ((split+round(lenLeft/2))-split)):

    if lenLeft%2 != 0:   #if the number of left patches are odd, jump one sample to make it even
        split = split+1

    tmp_nonmatches = np.concatenate(([imgsAdrsList[(split+round(lenLeft/2)):len(imgsAdrsList)]], [imgsAdrsList[split:(split+round(lenLeft/2))]]), axis=0)
    tmp_nonmatches = tmp_nonmatches.T
    nonmatches = np.zeros(tmp_nonmatches.size, dtype='U6')
    nonmatches = tmp_nonmatches

    print("\nCreated " + str(len(matches)) + " matches and " + str(len(nonmatches)) + " non-matches")

    return matches, nonmatches


def DBcombiner(DBs):
    AllMatches = np.array([[0,0]], dtype='U6')
    AllNonMatches = np.array([[0,0]], dtype='U6')
    for db in DBs:
        #making patch matches
        if os.path.exists(TEMPS_DIR+db+"_matches.csv"):  #read from saved files if already available
            print("skipping matchmaking process for " + db + " dataset")
            matchSet = np.genfromtxt(TEMPS_DIR+db+"_matches.csv", delimiter=',', dtype=None, encoding=None)
            nonmatchSet = np.genfromtxt(TEMPS_DIR+db+"_nonmatches.csv", delimiter=',', dtype=None, encoding=None)
        else:
            matchSet, nonmatchSet = matchMaker(DATA_DIR, db)    # o.w. compute
            matchSet = np.core.defchararray.add(DATA_DIR+db+'/', matchSet)
            nonmatchSet = np.core.defchararray.add(DATA_DIR+db+'/', nonmatchSet)
            np.savetxt(TEMPS_DIR+db+"_matches.csv", matchSet, delimiter=",", fmt='%s')
            np.savetxt(TEMPS_DIR+db+"_nonmatches.csv", nonmatchSet, delimiter=",", fmt='%s')
        
        # matches_path = os.path.join(TEMPS_DIR, (db+"_matches.csv"))
        # nonmatches_path = os.path.join(TEMPS_DIR, (db+"_nonmatches.csv"))
        # matches = np.genfromtxt(matches_path, delimiter=',', dtype=None, encoding=None)
        # nonmatches = np.genfromtxt(nonmatches_path, delimiter=',', dtype=None, encoding=None)

        AllMatches = np.concatenate((AllMatches, matchSet), axis=0)
        AllNonMatches = np.concatenate((AllNonMatches, nonmatchSet), axis=0)

    
    AllMatches = np.delete(AllMatches, 0, 0) #remove the first fake row
    AllNonMatches = np.delete(AllNonMatches, 0, 0) #remove the first fake row
    np.savetxt(TEMPS_DIR+"All_matches.csv", AllMatches, delimiter=",", fmt='%s')
    np.savetxt(TEMPS_DIR+"All_nonmatches.csv", AllNonMatches, delimiter=",", fmt='%s')


class patchSet(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        # self.imgsAdrsList = list(sorted(os.listdir(os.path.join(root, DB))))
        # self.labelsAdrs = os.path.join(root, (DB+"_labels.csv"))
        matches_path = os.path.join(root, ("All_matches.csv"))
        nonmatches_path = os.path.join(root, ("All_nonmatches.csv"))
        matches = np.genfromtxt(matches_path, delimiter=',', dtype=None, encoding=None)
        nonmatches = np.genfromtxt(nonmatches_path, delimiter=',', dtype=None, encoding=None)

        self.patchPairs = np.concatenate((matches, nonmatches),axis=0)
        self.labels = np.concatenate((np.ones(len(matches)), np.zeros(len(nonmatches))))


    def __getitem__(self, idx):
        
        # load images
        img_path1 = os.path.join(self.patchPairs[idx,0])
        img1 = Image.open(img_path1).convert('L')
        
        img_path2 = os.path.join(self.patchPairs[idx,1])
        img2 = Image.open(img_path2).convert('L')

        #pre-process (intensity normalization)
        
        img = np.concatenate((img1, img2), axis=1)

        # img = torch.as_tensor(imgs, dtype=torch.int32)
        target = self.labels[idx]
        target = torch.as_tensor(target, dtype=torch.long)

        img = self.transforms(img)
        
        return img, target

    def __len__(self):
        return len(self.patchPairs)

