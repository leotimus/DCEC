import os


def get_k_estimation(contig_file, hard=0):
    if os.path.exists(contig_file+'.seed'):
        seed_list = []
        with open(contig_file+'.seed') as f:
            for line in f:
                seed_list.append(line.rstrip('\n'))
        return seed_list

    dir_path = os.path.dirname(os.path.realpath(__file__))
    fragScanURL = os.path.join(
        dir_path, 'scripts', 'FragGeneScan1.19', 'run_FragGeneScan.pl')
    os.system("chmod 777 " + fragScanURL)
    hmmExeURL = os.path.join(dir_path, 'scripts',
                             'hmmer-3.1b1', 'bin', 'hmmsearch')
    os.system("chmod 777 " + hmmExeURL)
    markerExeURL = os.path.join(dir_path, 'scripts', 'test_getmarker.pl')
    os.system("chmod 777 " + markerExeURL)
    markerURL = os.path.join(os.getcwd(), 'scripts', 'marker.hmm')
    seedURL = contig_file+".seed"
    fragResultURL = contig_file+".frag.faa"
    hmmResultURL = contig_file+".hmmout"
    if not (os.path.exists(fragResultURL)):
        fragCmd = fragScanURL+" -genome="+contig_file+" -out="+contig_file + \
            ".frag -complete=0 -train=complete -thread=10 1>" + \
            contig_file+".frag.out 2>"+contig_file+".frag.err"
        print("exec cmd: "+fragCmd)
        os.system(fragCmd)

    if os.path.exists(fragResultURL):
        if not (os.path.exists(hmmResultURL)):
            hmmCmd = hmmExeURL+" --domtblout "+hmmResultURL+" --cut_tc --cpu 10 " + \
                markerURL+" "+fragResultURL+" 1>"+hmmResultURL+".out 2>"+hmmResultURL+".err"
            print("exec cmd: "+hmmCmd)
            os.system(hmmCmd)

        if os.path.exists(hmmResultURL):
            if not (os.path.exists(seedURL)):
                markerCmd = markerExeURL+" "+hmmResultURL+" " + \
                    contig_file+" 1001 "+seedURL+" "+str(hard)
                print("exec cmd: "+markerCmd)
                os.system(markerCmd)

            if os.path.exists(seedURL):
                seed_list = []
                with open(seedURL) as f:
                    for line in f:
                        seed_list.append(line.rstrip('\n'))
                return seed_list
            else:
                return None
        else:
            print("Hmmsearch failed! Not exist: "+hmmResultURL)
    else:
        print("FragGeneScan failed! Not exist: "+fragResultURL)