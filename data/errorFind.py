import csv
import sys
from glob import glob
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from natsort import natsorted as nsd
from Levenshtein import distance as dst
import numpy as np

def min_diff(text, queries):
    dffs = [dst(text, x[1]) for x in queries]
    idx = dffs.index(min(dffs))
    if dffs[idx] > len(text)/3:
        return -1
    return idx

def read_both(gt, pr):
    with open(gt, "r") as fr:
        lines_gt = [(x[-1], x[-2][1:], [int(y) for y in x[1:9]]) for x in csv.reader(fr)]
    lines_gt.sort(key = lambda x: x[1])

    with open(pr, "r") as fr:
        lines_pr = [(x[0], x[1][1:]) for x in csv.reader(fr, delimiter='\t')]
    lines_pr.sort(key = lambda x: x[1])

    i = 0
    j = 0
    vrd = []
    for tag, text, cords in lines_gt:
        md = min_diff(text, lines_pr)
        if md == -1:
            vrd.append((text, tag, "", cords))
        else:
            vrd.append((text, tag, lines_pr[md][0], cords))

    return vrd

def fnd(txts, type):
    if type == "R":
        return min_diff("REPÚBLICA FEDERATIVA DO BRASIL", txts)
    elif type == "C":
        return min_diff("CARTEIRA DE IDENTIDADE", txts)

def direction(gts, txts, cds, cl, f1n):
    # step one: verify type.
    if "CNH" in cl:
        if "Aberta" in cl or "Frente" in cl:
            hf = cds[gts.index("info-nome")]
            hs = cds[gts.index("info-primeirahab")]
        else:
            hf = cds[gts.index("info-obs")]
            hs = cds[gts.index("info-dataemissao")]
    else:
        if "Frente" in cl:
            idx1 = fnd(txts, "R")
            idx2 = fnd(txts, "C")
            hf = cds[idx1]
            hs = cds[idx2]
        else:
            hf = cds[gts.index("info-nome")]
            hs = cds[gts.index("info-datanasc")]

    # step two: verify direction.
    if hf[0] > hs[0]:
        if hf[1] > hs[1]:
            dir = "L"
        else:
            dir = "U"
    elif hf[1] < hs[1]:
        dir = "D"
    else:
        dir = "R"

    # step three: normalize.
    im = Image.open(f1n)
    wd, hg = im.size
    im.close()
    if dir == "D":
        for cd in cds:
            cd[0] = int((-(cd[0] - wd/2)) + wd/2)
            cd[1] = int((-(cd[1] - hg/2)) + hg/2)
            cd[4] = int((-(cd[4] - wd/2)) + wd/2)
            cd[5] = int((-(cd[5] - hg/2)) + hg/2)
    elif dir == "R":
        for cd in cds:
            cd[0] = int(((cd[1] - wd/2)) + wd/2)
            cd[1] = int((-(cd[0] - hg/2)) + hg/2)
            cd[4] = int(((cd[5] - wd/2)) + wd/2)
            cd[5] = int((-(cd[4] - hg/2)) + hg/2)
    elif dir == "L":
        for cd in cds:
            cd[0] = int((-(cd[1] - hg/2)) + hg/2)
            cd[1] = int(((cd[0] - wd/2)) + wd/2)
            cd[4] = int((-(cd[5] - hg/2)) + hg/2)
            cd[5] = int(((cd[4] - wd/2)) + wd/2)
    return cds

def compare(f1, f2, cl):
    global astm, aste, conf_matrix
    f1n = f1[:]

    res = read_both(f1, f2)
    texts = [x[0] for x in res]
    gts = [x[1] for x in res]
    prs = [x[2] for x in res]
    cds = [x[3] for x in res]
    if cl != "CNH" and cl != "RG":
        cds = direction(gts, texts, cds, cl, f"./{cl}/{f1n.split('/')[-1].replace('tsv', 'jpg')}")

    hit = 0
    miss = 0
    err = 0

    for text, gt, pr, cs in zip(texts, gts, prs, cds):
        if pr != "":
            if gt not in conf_matrix:
                conf_matrix[gt] = {}
            if pr not in conf_matrix[gt]:
                conf_matrix[gt][pr] = 0
            conf_matrix[gt][pr] += 1
        if gt == pr:
            hit += 1
        elif pr == "":
            miss += 1
            if gt not in astm:
                astm[gt] = (np.zeros((1000, 1000)), [])
            astm[gt][0][cs[0]:cs[4],cs[1]:cs[5]] += 1
            if pr not in astm[gt][1]:
                astm[gt][1].append(f2)
        else:
            err += 1
            if gt not in aste:
                aste[gt] = (np.zeros((1000, 1000)), [])
            aste[gt][0][cs[0]:cs[4],cs[1]:cs[5]] += 1
            if pr not in aste[gt][1]:
                aste[gt][1].append(f2)

    return hit, miss, err

def get_file_names(csv_file, classe=""):
    with open(csv_file, "r") as cr:
        ret = [x[-1].replace("jpg", "tsv") for x in csv.reader(cr) if classe in x[1]]
    return ret

def main(argv, classe="", md="w", n_it='4'):
    global conf_matrix
    conf_matrix = {}
    global astm, aste
    astm = {}
    aste = {}

    global mss, irr, gterr, prerr
    mss = {}
    gterr = {}
    prerr = {}
    irr = []
    fs = get_file_names(argv[0], classe=classe)
    gt = [f"./t1/" + x for x in fs]
    pred = [f"./out{n_it}/" + x.replace(".tsv", ".txt") for x in fs]
    # n_it = "4"
    with open(f"./aux{n_it}/errorLog_{classe}.txt", "w") as logger:
        total_hit = 0
        total_miss = 0
        total_err = 0
        perf = 0
        missing = 0
        mss_files = []
        err_files = []
        for g, p in zip(gt, pred):
            hit, miss, err = compare(g, p, classe)
            if miss != 0:
                missing += 1
                mss_files.append(p)
            elif err == 0:
                perf += 1
            else:
                err_files.append(p)
            total_hit += hit
            total_miss += miss
            total_err += err

        logger.write(f"{classe}:\n\n{len(gt)} imagens com {total_hit + total_miss + total_err} atributos ({(total_hit + total_miss + total_err)/len(gt)} atributos em média).\n")
        logger.write(f"{1 - (len(gt) - perf)/len(gt)}% ({perf}) imagens perfeitas - sem erros\n")
        logger.write(f"{1 - (len(gt) - missing)/len(gt)}% ({missing}) tiveram pelo menos um atributo não identificado.\n")
        logger.write(f"total accuracy: {total_hit/(total_hit + total_miss + total_err)}.\n")
        logger.write(f"total miss: {total_miss/(total_hit + total_miss + total_err)}.\n")
        logger.write(f"total err: {total_err/(total_hit + total_miss + total_err)}.\n")
    print(conf_matrix)

    for k, v in astm.items():
        new =  v[0] * (255.0/v[0].max())
        Image.fromarray(new).convert("L").save(f"./aux{n_it}/{k}_mss.png")
        with open(f"./aux{n_it}/{k}_mss.txt", md) as lg:
            lg.write(f"{v[1]}")
    for k, v in aste.items():
        new = v[0] * (255.0/v[0].max())
        Image.fromarray(new).convert("L").save(f"./aux{n_it}/{k}_err.png")
        with open(f"./aux{n_it}/{k}_err.txt", md) as lg:
            lg.write(f"{v[1]}")

    its = list(conf_matrix.keys())
    n_its = len(its)
    bij = {k: v for (k, v) in zip(its, range(n_its))}
    conf_img = np.zeros((n_its*50, n_its*50, 3), dtype=np.uint8)
    bp = Image.new('RGB', conf_img.shape[:-1])
    drawer = ImageDraw.Draw(bp)

    for i in range(n_its):
        total = 0
        rems = []
        idx_x = i*50
        drawer.multiline_text((0, idx_x + 50),its[i].replace('-', '\n'))
        for j in conf_matrix[its[i]].keys():
            if j not in bij:
                rems.append(j)
            elif its[i] != j:
                total += conf_matrix[its[i]][j]
        for j in rems:
            del conf_matrix[its[i]][j]
        for j in conf_matrix[its[i]].keys():
            idx_y = bij[j]*50
            drawer.text((idx_y + 50, 0),j.replace('-', '\n'))
            if j == its[i]:
                conf_img[idx_x:50 + idx_x, idx_y:50 + idx_y, 2] += int(((conf_matrix[its[i]][j]*255)/(total + conf_matrix[its[i]][j])))
            else:
                conf_img[idx_x:50 + idx_x, idx_y:50 + idx_y, 0] += int(((conf_matrix[its[i]][j]*255)/total))
    img = Image.fromarray(conf_img,mode='RGB')
    bp.paste(img, box=(50, 50))
    for i in range(n_its):
        idx_x = i*50 + 65
        for j in range(n_its):
            idx_y = j*50 + 65
            if its[j] in conf_matrix[its[i]]:
                drawer.text((idx_y, idx_x),str(conf_matrix[its[i]][its[j]]))
            else:
                drawer.text((idx_y, idx_x),'0')


    bp.save(f"./aux{n_it}/conf_{classe}.png")

    return

if __name__ == "__main__":
    # for cl in ["CNH_Frente", "CNH_Verso", "CNH_Aberta"]:
    #     main(sys.argv[1:], classe=cl, mode="w" if cl == "CNH_Aberta" else "a", n_it='4')
    for cl in ["RG_Frente", "RG_Verso", "RG_Aberto"]:
        if cl == "RG_Frente":
            m = "w"
        else:
            m = "a"
        main(sys.argv[1:], classe=cl, md=m, n_it='3')
