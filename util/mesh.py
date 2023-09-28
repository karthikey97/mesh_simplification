import numpy as np
import numpy.linalg as la
from pytorch3d.io import load_objs_as_meshes
import scipy as sp
import heapq
import copy
from sklearn.preprocessing import normalize
import time

OPTIM_VALENCE = 6
VALENCE_WEIGHT = 1

class Mesh:
    def __init__(self, path):
        self.path = path
        mesh = load_objs_as_meshes([path])
        # self.vs, self.faces = self.fill_from_file(path)
        self.vs = mesh.verts_packed().numpy()
        self.faces = mesh.faces_packed().numpy()
        self.edges = mesh.edges_packed().numpy()
        self.fn = mesh.faces_normals_packed().numpy()
        self.fc = np.sum(self.vs[self.faces], 1) / 3.0
        self.fa = mesh.faces_areas_packed().numpy()
        self.get_abcd()
        self.build_v2v()
        self.build_vf()
        self.simp = False

    # def compute_face_normals(self):
    #     face_normals = np.cross(self.vs[self.faces[:, 1]] - self.vs[self.faces[:, 0]], self.vs[self.faces[:, 2]] - self.vs[self.faces[:, 0]])
    #     norm = np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-24
    #     face_areas = 0.5 * np.sqrt((face_normals**2).sum(axis=1))
    #     face_normals /= norm
    #     self.fn, self.fa = face_normals, face_areas

    def get_abcd(self):
        self.p = self.vs[self.faces[:, 0]]
        self.e1 = self.vs[self.faces[:, 1]] - self.p
        self.e1 /= la.norm(self.e1, axis=1, keepdims=True) + 1e-24

        e2 = self.vs[self.faces[:, 2]] - self.p
        e2_p_e1 = (e2*self.e1).sum(axis=1, keepdims=True)*self.e1
        self.e2 = e2 - e2_p_e1
        self.e2 /= la.norm(self.e2, axis=1, keepdims=True) + 1e-24

        # d_s = - 1.0 * np.sum(self.fn * self.fc, axis=1, keepdims=True)
        # self.abcd = np.concatenate([self.fn, d_s], axis=1)

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

    def simplification(self, target_v):
        vs, vf, edges = self.vs, self.vf, self.edges
        ndim = vs.shape[1]

        """ 1. compute Q for each vertex """
        self.A_s = [np.zeros((3,3)) for _ in range(len(vs))]
        self.B_s = [np.zeros((3,1)) for _ in range(len(vs))]
        self.C_s = [np.zeros((1,1)) for _ in range(len(vs))]
        # self.Q_s = [[] for _ in range(len(vs))]
        for i, v in enumerate(vs):
            f_s = list(vf[i])
            for f in f_s:
                e1, e2, p = self.e1[f][:,None], self.e2[f][:,None], self.p[f][:,None]
                
                pe1 = np.sum(p * e1)
                pe2 = np.sum(p * e2)

                self.A_s[i] += np.eye(ndim)-np.matmul(e1, e1.T)-np.matmul(e2,e2.T)
                self.B_s[i] += e1*pe1 + e2*pe2 - p
                self.C_s[i] += np.matmul(p.T, p) - pe1**2 - pe2**2

                # E_a = np.matmul(p.T,np.matmul(A, p)) 
                # E_b = 2*np.matmul(b.T, v) 
                # E_c = c
                # print(E_a + E_b + E_c)

            # e1_s, e2_s, p_s = self.e1[f_s[0:1]], self.e2[f_s[0:1]], self.p[f_s[0:1]]
            # pe1 = np.sum(p_s * e1_s, axis=1, keepdims=True)
            # pe2 = np.sum(p_s * e2_s, axis=1, keepdims=True)

            # self.A_s[i] = np.eye(ndim)*len(f_s) - np.matmul(e1_s.T, e1_s) - np.matmul(e2_s.T, e2_s)
            # self.B_s[i] = np.matmul(e1_s.T, pe1) + np.matmul(e2_s.T, pe2) - np.sum(p_s.T, axis=1, keepdims=True)
            # self.C_s[i] = np.sum(p_s**2 - pe1**2 - pe2**2)

            # E_a = np.matmul(v.T,np.matmul(self.A_s[i], v)) 
            # E_b = 2*np.matmul(self.B_s[i].T, v)[0]
            # E_c = self.C_s[i][0,0]

            # f_s = np.array(list(vf[i]))
            # abcd_s = self.abcd[f_s]
            # self.Q_s[i] = np.matmul(abcd_s.T, abcd_s)
            # v4 = np.concatenate([v, np.array([1])])
            # E_s = np.matmul(v4, np.matmul(self.Q_s[i], v4.T))
            
            # print(E_a+E_b+E_c)

        """ 2. compute E for every possible pairs and create heapq """
        E_heap = []
        for i, e in enumerate(edges):
            v_0, v_1 = vs[e[0]], vs[e[1]]
            A_new, B_new, C_new = (self.A_s[e[0]]+self.A_s[e[1]]),(self.B_s[e[0]]+self.B_s[e[1]]),(self.C_s[e[0]]+self.C_s[e[1]])

            try:
                v_new = -np.matmul(la.inv(A_new), B_new)
            except:
                v_new = (0.5 * (v_0 + v_1))[:,None]

            # E_new = np.matmul(v_new, B_new)[0] + C_new
            E_new = np.matmul(v_new.T,np.matmul(A_new, v_new)) + 2*np.matmul(B_new.T, v_new) + C_new
            # print(E_new)
            heapq.heappush(E_heap, (E_new[0,0], (e[0], e[1]), v_new[:,0], (A_new, B_new, C_new)))
        """ 3. collapse minimum-error vertex """
        simp_mesh = copy.deepcopy(self)

        vi_mask = np.ones([len(simp_mesh.vs)]).astype(np.bool_)
        fi_mask = np.ones([len(simp_mesh.faces)]).astype(np.bool_)

        vert_map = [{i} for i in range(len(simp_mesh.vs))]

        while np.sum(vi_mask) > target_v:
            if len(E_heap) == 0:
                print("[Warning]: edge cannot be collapsed anymore!")
                break

            E_0, (vi_0, vi_1), v_new, Q_new = heapq.heappop(E_heap)

            if (vi_mask[vi_0] == False) or (vi_mask[vi_1] == False):
                continue

            """ edge collapse """
            shared_vv = list(set(simp_mesh.v2v[vi_0]).intersection(set(simp_mesh.v2v[vi_1])))
            merged_faces = simp_mesh.vf[vi_0].intersection(simp_mesh.vf[vi_1])

            if len(shared_vv) != 2:
                """ non-manifold! """
                # print("non-manifold can be occured!!" , len(shared_vv))
                continue

            elif len(merged_faces) != 2:
                """ boundary """
                # print("boundary edge cannot be collapsed!")
                continue

            else:
                self.edge_collapse(simp_mesh, vi_0, vi_1, merged_faces, vi_mask, fi_mask, vert_map, v_new, Q_new, E_heap)
                # print(np.sum(vi_mask), np.sum(fi_mask))
        
        self.rebuild_mesh(simp_mesh, vi_mask, fi_mask, vert_map)
        simp_mesh.simp = True
        self.build_hash(simp_mesh, vi_mask, vert_map)
        
        return simp_mesh

    def edge_collapse(self, simp_mesh, vi_0, vi_1, merged_faces, vi_mask, fi_mask, vert_map, v_new, Q_new, E_heap):
        shared_vv = list(set(simp_mesh.v2v[vi_0]).intersection(set(simp_mesh.v2v[vi_1])))
        new_vi_0 = set(simp_mesh.v2v[vi_0]).union(set(simp_mesh.v2v[vi_1])).difference({vi_0, vi_1})
        simp_mesh.vf[vi_0] = simp_mesh.vf[vi_0].union(simp_mesh.vf[vi_1]).difference(merged_faces)
        simp_mesh.vf[vi_1] = set()
        simp_mesh.vf[shared_vv[0]] = simp_mesh.vf[shared_vv[0]].difference(merged_faces)
        simp_mesh.vf[shared_vv[1]] = simp_mesh.vf[shared_vv[1]].difference(merged_faces)

        simp_mesh.v2v[vi_0] = list(new_vi_0)
        for v in simp_mesh.v2v[vi_1]:
            if v != vi_0:
                simp_mesh.v2v[v] = list(set(simp_mesh.v2v[v]).difference({vi_1}).union({vi_0}))
        simp_mesh.v2v[vi_1] = []
        vi_mask[vi_1] = False

        vert_map[vi_0] = vert_map[vi_0].union(vert_map[vi_1])
        vert_map[vi_0] = vert_map[vi_0].union({vi_1})
        vert_map[vi_1] = set()
        
        fi_mask[np.array(list(merged_faces)).astype(np.int64)] = False

        # simp_mesh.vs[vi_0] = 0.5 * (simp_mesh.vs[vi_0] + simp_mesh.vs[vi_1])
        simp_mesh.vs[vi_0] = v_new
        self.A_s[vi_0], self.B_s[vi_0], self.C_s[vi_0] = Q_new

        """ recompute E """
        for vv_i in simp_mesh.v2v[vi_0]:
            A_new, B_new, C_new = (self.A_s[vi_0]+self.A_s[vv_i]),(self.B_s[vi_0]+self.B_s[vv_i]),(self.C_s[vi_0]+self.C_s[vv_i])
            # v_mid = 0.5 * (simp_mesh.vs[vi_0] + simp_mesh.vs[vv_i])[:,None]
            try:
                v_mid = -np.matmul(la.inv(A_new), B_new).reshape(-1)
            except:
                v_mid = 0.5 * (simp_mesh.vs[vi_0] + simp_mesh.vs[vv_i])[:,None]

            E_new = np.matmul(v_mid.T,np.matmul(A_new, v_mid)) + 2*np.matmul(B_new.T, v_mid) + C_new
            heapq.heappush(E_heap, (E_new[0,0], (vi_0, vv_i), v_mid, (A_new, B_new, C_new)))

    @staticmethod
    def rebuild_mesh(simp_mesh, vi_mask, fi_mask, vert_map):
        face_map = dict(zip(np.arange(len(vi_mask)), np.cumsum(vi_mask)-1))
        simp_mesh.vs = simp_mesh.vs[vi_mask]
        
        vert_dict = {}
        for i, vm in enumerate(vert_map):
            for j in vm:
                vert_dict[j] = i

        for i, f in enumerate(simp_mesh.faces):
            for j in range(3):
                if f[j] in vert_dict:
                    simp_mesh.faces[i][j] = vert_dict[f[j]]

        simp_mesh.faces = simp_mesh.faces[fi_mask]
        for i, f in enumerate(simp_mesh.faces):
            for j in range(3):
                simp_mesh.faces[i][j] = face_map[f[j]]
        
        # simp_mesh.compute_face_normals()
        # simp_mesh.compute_face_center()
        # simp_mesh.build_gemm()
        # simp_mesh.compute_vert_normals()
        # simp_mesh.build_v2v()
        # simp_mesh.build_vf()

    @staticmethod
    def build_hash(simp_mesh, vi_mask, vert_map):
        pool_hash = {}
        unpool_hash = {}
        for simp_i, idx in enumerate(np.where(vi_mask)[0]):
            if len(vert_map[idx]) == 0:
                print("[ERROR] parent node cannot be found!")
                return
            for org_i in vert_map[idx]:
                pool_hash[org_i] = simp_i
            unpool_hash[simp_i] = list(vert_map[idx])
        
        """ check """
        vl_sum = 0
        for vl in unpool_hash.values():
            vl_sum += len(vl)

        if (len(set(pool_hash.keys())) != len(vi_mask)) or (vl_sum != len(vi_mask)):
            print("[ERROR] Original vetices cannot be covered!")
            return
        
        pool_hash = sorted(pool_hash.items(), key=lambda x:x[0])
        simp_mesh.pool_hash = pool_hash
        simp_mesh.unpool_hash = unpool_hash
            
    def save(self, filename):
        assert len(self.vs) > 0
        vertices = np.array(self.vs, dtype=np.float32).flatten()
        indices = np.array(self.faces, dtype=np.uint32).flatten()

        with open(filename, 'w') as fp:
            # Write positions
            for i in range(0, vertices.size, 3):
                x = vertices[i + 0]
                y = vertices[i + 1]
                z = vertices[i + 2]
                fp.write('v {0:.8f} {1:.8f} {2:.8f}\n'.format(x, y, z))

            # Write indices
            for i in range(0, len(indices), 3):
                i0 = indices[i + 0] + 1
                i1 = indices[i + 1] + 1
                i2 = indices[i + 2] + 1
                fp.write('f {0} {1} {2}\n'.format(i0, i1, i2))
