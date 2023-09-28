import argparse
import os
import time

from util.mesh import Mesh

def get_parser():
    parser = argparse.ArgumentParser(description="Mesh Simplification")
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("--v", type=int, required=True)
    args = parser.parse_args()
    return args

def main():
    args = get_parser()
    mesh = Mesh(args.input)
    mesh_name = os.path.basename(args.input).split(".")[-2]
    if args.v >= mesh.vs.shape[0]:
        print("[ERROR]: Target vertex number should be smaller than {}!".format(mesh.vs.shape[0]))
        exit()
    tx = time.time()
    simp_mesh = mesh.simplification(target_v=args.v)
    print(time.time()-tx)
    # simp_mesh = mesh.edge_based_simplification(target_v=args.v, valence_aware=args.optim)
    os.makedirs("data/output/", exist_ok=True)
    simp_mesh.save("data/output/{}_{}.obj".format(mesh_name, simp_mesh.vs.shape[0]))
    print("[FIN] Simplification Completed!")

if __name__ == "__main__":
    main()