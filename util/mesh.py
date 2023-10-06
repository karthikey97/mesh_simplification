import torch
import numpy as np
import numpy.linalg as la
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.renderer import TexturesUV
import heapq
from bisect import insort
import copy


OPTIM_VALENCE = 6
VALENCE_WEIGHT = 1

class Mesh:
    def __init__(self, path):
        self.path = path
        mesh = load_objs_as_meshes([path])
        textures = mesh.textures

        verts = mesh.verts_packed().numpy()
        faces = mesh.faces_packed().numpy()
        ft = textures.faces_uvs_padded().numpy()[0,:,:]
        vt = textures.verts_uvs_padded().numpy()[0,:,:]
        self.texture_map = textures.maps_padded()[0]

        self.clean_mesh(faces, verts, ft, vt)
        self.ndim = self.vs.shape[1]

        self.simp = False

    def clean_mesh(self, faces, verts, ft, vt):
        fvt = np.stack([faces,ft], axis=-1).reshape(-1, 2)
        fvs = np.concatenate([verts[faces], vt[ft]], axis=-1).reshape(fvt.shape[0], -1)

        unique_pairs, indx, inv = np.unique(fvt, axis=0, return_index=True, return_inverse=True)
        self.faces = inv.reshape(faces.shape)
        self.vs = fvs[indx]
        
        self.vert_mapping = [[] for _ in range(len(verts))]
        for i,new_vs in enumerate(unique_pairs[:,0]):
            self.vert_mapping[new_vs].append(i)
        
        edges, e2k, ec = [], {}, 0
        for f in self.faces:
            for i in range(3):
                e = tuple(sorted([f[i], f[(i+1)%3]]))
                if e not in e2k:
                    e2k[e] = ec
                    ec+=1
                    edges.append(list(e))
        self.edges = self.edges = np.array(edges, dtype=np.int64)

    def build_vf(self):
        vf = [set() for _ in range(len(self.vs))]
        for i, f in enumerate(self.faces):
            vf[f[0]].add(i)
            vf[f[1]].add(i)
            vf[f[2]].add(i)
        self.vf = vf

    def build_v2v(self):
        v2v = [[] for _ in range(len(self.vs))]
        for i, e in enumerate(self.edges):
            v2v[e[0]].append(e[1])
            v2v[e[1]].append(e[0])
        self.v2v = v2v

    def get_abc(self):
        self.p = self.vs[self.faces[:, 0]]
        self.e1 = self.vs[self.faces[:, 1]] - self.p
        self.e1 /= la.norm(self.e1, axis=1, keepdims=True) + 1e-24

        e2 = self.vs[self.faces[:, 2]] - self.p
        e2_p_e1 = (e2*self.e1).sum(axis=1, keepdims=True)*self.e1
        self.e2 = e2 - e2_p_e1
        self.e2 /= la.norm(self.e2, axis=1, keepdims=True) + 1e-24

        self.A_s = [np.zeros((self.ndim,self.ndim)) for _ in range(len(self.vs))]
        self.B_s = [np.zeros((self.ndim,1)) for _ in range(len(self.vs))]
        self.C_s = [np.zeros((1,1)) for _ in range(len(self.vs))]

        for i, v in enumerate(self.vs):
            f_s = list(self.vf[i])
            nf = len(self.vf[i])
            for f in f_s:
                e1, e2, p = self.e1[f][:,None], self.e2[f][:,None], self.p[f][:,None]
                
                pe1 = np.sum(p * e1)
                pe2 = np.sum(p * e2)

                self.A_s[i] += np.eye(self.ndim)-np.matmul(e1, e1.T)-np.matmul(e2,e2.T)
                self.B_s[i] += e1*pe1 + e2*pe2 - p
                self.C_s[i] += np.matmul(p.T, p) - pe1**2 - pe2**2
            self.A_s[i] /= nf
            self.B_s[i] /= nf
            self.C_s[i] /= nf
            # E = np.matmul(v.T,np.matmul(self.A_s[i], v)) + 2*np.matmul(self.B_s[i].T, v) + self.C_s[i]

    def build_boundary_quadratics(self, boundary_weight=100):
        ef = [[] for _ in range(len(self.edges))]
        boundary_edge_count = 0
        for i, (v0,v1) in enumerate(self.edges):
            ef[i] = self.vf[v0].intersection(self.vf[v1])
            if len(ef[i])==1:
                boundary_edge_count += 1
                f = self.faces[ef[i].pop()]
                v2 = (set(f).difference(set([v0,v1]))).pop()
                p,q,r = self.vs[v0], self.vs[v1], self.vs[v2]
                n = (r-p) - ((r-p)*(q-p)).sum()*(q-p)/la.norm(q-p)**2
                n /= la.norm(n)
                pn = (p*n).sum()
                A = boundary_weight*np.matmul(n[:,None],n[:,None].T)
                B = -boundary_weight*pn*n[:,None]
                C = boundary_weight*pn**2
                self.A_s[v0] += A
                self.A_s[v1] += A
                self.B_s[v0] += B
                self.B_s[v1] += B
                self.C_s[v0] += C
                self.C_s[v1] += C
                
    def simplification(self, target_f, boundary_weight=100):
        self.build_v2v()
        self.build_vf()
        self.get_abc()
        self.build_boundary_quadratics(boundary_weight)

        """ Compute E for every possible pairs and create heapq """
        E_heap = []
        for i, e in enumerate(self.edges):
            v_0, v_1 = self.vs[e[0]], self.vs[e[1]]
            A_new,B_new,C_new = self.A_s[e[0]]+self.A_s[e[1]], self.B_s[e[0]]+self.B_s[e[1]], self.C_s[e[0]]+self.C_s[e[1]]

            try:
                v_new = -np.matmul(la.inv(A_new), B_new)
            except:
                v_new = (0.5 * (v_0 + v_1))[:,None]

            E_new = np.matmul(v_new.T,np.matmul(A_new, v_new)) + 2*np.matmul(B_new.T, v_new) + C_new
            E_heap.append((E_new[0,0], (e[0], e[1]), v_new[:,0]))
        E_heap.sort()

        """ Collapse minimum-error vertex """
        simp_mesh = copy.deepcopy(self)
        vi_mask = np.ones([len(simp_mesh.vs)]).astype(np.bool_)
        fi_mask = np.ones([len(simp_mesh.faces)]).astype(np.bool_)
        vert_map = [{i} for i in range(len(simp_mesh.vs))]
        while np.sum(fi_mask) > target_f and len(E_heap)>0:
            E_0, (vi_0, vi_1), v_new = E_heap.pop(0)
            if (vi_mask[vi_0] == False) or (vi_mask[vi_1] == False):
                continue
            
            """ edge collapse """
            merged_faces = simp_mesh.vf[vi_0].intersection(simp_mesh.vf[vi_1])
            simp_mesh.vs[vi_0] = v_new
            vi_mask[vi_1] = False
            fi_mask[np.array(list(merged_faces)).astype(np.int64)] = False
            vert_map[vi_0] = vert_map[vi_0].union(vert_map[vi_1])
            vert_map[vi_0] = vert_map[vi_0].union({vi_1})
            vert_map[vi_1] = set()
            self.edge_collapse(simp_mesh, (vi_0, vi_1), E_heap)
        
        self.rebuild_mesh(simp_mesh, vi_mask, fi_mask, vert_map)
        return simp_mesh

    @staticmethod
    def edge_collapse(simp_mesh, edge, E_heap):
        vi_0, vi_1 = edge
        merged_faces = simp_mesh.vf[vi_0].intersection(simp_mesh.vf[vi_1])
        shared_vv = list(set(simp_mesh.v2v[vi_0]).intersection(set(simp_mesh.v2v[vi_1])))
        new_vi_0 = set(simp_mesh.v2v[vi_0]).union(set(simp_mesh.v2v[vi_1])).difference({vi_0, vi_1})
        simp_mesh.vf[vi_0] = simp_mesh.vf[vi_0].union(simp_mesh.vf[vi_1]).difference(merged_faces)
        simp_mesh.vf[vi_1] = set()
        for vv in shared_vv:
            simp_mesh.vf[vv] = simp_mesh.vf[vv].difference(merged_faces)

        simp_mesh.v2v[vi_0] = list(new_vi_0)
        for v in simp_mesh.v2v[vi_1]:
            if v != vi_0:
                simp_mesh.v2v[v] = list(set(simp_mesh.v2v[v]).difference({vi_1}).union({vi_0}))
        simp_mesh.v2v[vi_1] = []

        
        invalid_heap_idxs = [i for i,E in enumerate(E_heap) if vi_0 in E[1] or vi_1 in E[1]]
        invalid_heap_idxs.sort(reverse=True)
        for idx in invalid_heap_idxs: E_heap.pop(idx)

        """ recompute E """
        for vv_i in simp_mesh.v2v[vi_0]:
            A_new = simp_mesh.A_s[vi_0]+simp_mesh.A_s[vv_i]
            B_new = simp_mesh.B_s[vi_0]+simp_mesh.B_s[vv_i]
            C_new = simp_mesh.C_s[vi_0]+simp_mesh.C_s[vv_i]
            try:
                v_mid = -np.matmul(la.inv(A_new), B_new).reshape()
            except:
                v_mid = 0.5 * (simp_mesh.vs[vi_0] + simp_mesh.vs[vv_i])[:,None]

            E_new = np.matmul(v_mid.T,np.matmul(A_new, v_mid)) + 2*np.matmul(B_new.T, v_mid) + C_new
            insort(E_heap, (E_new[0,0], (vi_0, vv_i), v_mid[:,0]))

    @staticmethod
    def rebuild_mesh(simp_mesh, vi_mask, fi_mask, vert_map):
        vert_dict = {}
        for i, vm in enumerate(vert_map):
            for j in vm:
                vert_dict[j] = i
        
        simp_mesh.verts = simp_mesh.vs[:,:3]
        simp_mesh.vt = simp_mesh.vs[:,3:5]

        # for vmap in simp_mesh.vert_mapping:
        #     if len(vmap)>1:
        #         a = []
        #         for v in vmap: a.append(vert_dict[v])
        #     v_corrected = np.zeros(3)
        #     c=0
        #     for v in vmap:
        #         v_corrected += simp_mesh.verts[vert_dict[v]]
        #         c+=1
        #     for v in vmap:
        #         simp_mesh.verts[vert_dict[v]] = v_corrected/c

        face_map = dict(zip(np.arange(len(vi_mask)), np.cumsum(vi_mask)-1))
        simp_mesh.verts = simp_mesh.verts[vi_mask]
        simp_mesh.vt = simp_mesh.vt[vi_mask]
        simp_mesh.vs = simp_mesh.vs[vi_mask]

        for i, f in enumerate(simp_mesh.faces):
            for j in range(3):
                if f[j] in vert_dict:
                    simp_mesh.faces[i][j] = vert_dict[f[j]]

        simp_mesh.faces = simp_mesh.faces[fi_mask]
        for i, f in enumerate(simp_mesh.faces):
            for j in range(3):
                simp_mesh.faces[i][j] = face_map[f[j]]
        
        simp_mesh.simp = True

    def save(self,filename):
        vertices = torch.tensor(self.verts)
        verts_uvs = torch.tensor(self.vt.clip(0.,1.))
        faces = torch.tensor(self.faces)
        faces_uvs = torch.tensor(self.faces)
        save_obj(filename, verts=vertices, faces=faces, verts_uvs=verts_uvs, faces_uvs=faces_uvs, texture_map=self.texture_map)
