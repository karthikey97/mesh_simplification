# メッシュ簡略化
"Surface Simplification using Quadric Error Metrics, 1997" [[Paper]](http://www.cs.cmu.edu/~garland/Papers/quadrics.pdf) を実装。

## アルゴリズム

### 概要
頂点 $\bm{v}=(v_x, v_y, v_z, 1)^T$ 毎のコストを、 $4\times4$ 対称行列 $Q$ を用いて、二次形式 $\Delta(\bm{v})=\bm{v}^T Q \bm{v}$ で定義。
縮約した場合に、コストが最小となる頂点ペアから縮約する。

### 手順
1. 初期頂点で対称行列 $Q$ を計算する（後述）
2. 縮約できる頂点ペアをリストアップする
3. 2.の各頂点ペアに対し、縮約した場合のコストを計算する
   - 頂点 $\bm{v}_1$ を $\bm{v}_2$ にマージする場合、生成される頂点を $\bm{\bar{v}}=\frac{1}{2}(\bm{v}_1+\bm{v}_2)$ として、
   $\bm{\bar{v}}^T (Q_1+Q_2) \bm{\bar{v}}$ を頂点ペア $(\bm{v}_1, \bm{v}_2)$ のコストとする。
4. 3.で計算した各頂点ペアのコストを格納するヒープを作成
5. ヒープから最小コストとなる頂点ペア $(\bm{v}_1, \bm{v}_2)$ を取り出し、そのエッジを縮約する
   - この際、頂点 $\bm{v}_1$ が関与する全ての頂点ペアに対するコストを更新する
  
### $Q$ の定義

頂点周りの平面（三角形）を表す方程式を、$ax+by+cz+d=0$ とする。ただし、$a^2+b^2+c^2=1$ である。
すなわち、$(a, b, c)^T$ は三角形の法線ベクトルを意味し、三角形の重心座標を $(c_x, c_y, c_z)^T$ とすると、
$$ d = -1 \times
\left[ 
\begin{matrix}
a\\
b\\
c\\
\end{matrix}
\right]
\cdot
\left[ 
\begin{matrix}
c_x\\
c_y\\
c_z\\
\end{matrix}
\right]
$$
と表せる。
$\bm{p}=(a,b,c,d)^T$ として、
頂点 $\bm{v}$ から周囲の平面（三角形）$\bm{p}$ までの距離は
$$
\bm{p}^T \bm{v} = a v_x+ b v_y + c v_z + d
$$
と表現でき、これらの二乗誤差の総和は、
$$
\begin{align}
\Delta(\bm{v}) =& \sum_{\bm{p} \in N(\bm{v})}(\bm{p}^T \bm{v})^2 \\
=& \sum_{\bm{p} \in N(\bm{v})}(\bm{v}^T \bm{p})(\bm{p}^T \bm{v}) \\
=& \bm{v}^T \left(\sum_{\bm{p} \in N(\bm{v})}\bm{p}\bm{p}^T \right) \bm{v} \\
\end{align}
$$
となる。ここで、
$$ K_p = \bm{p}\bm{p}^T =
\left[
\begin{matrix} 
a^2 & ab & ac & ad \\ 
ab & b^2 & bc & bd \\
ac & bc & c^2 & cd \\
ad & bd & cd & d^2  
\end{matrix} 
\right]
$$
$$
Q = \sum_{\bm{p} \in N(\bm{v})} K_p
$$
と定義することで、頂点のコストを二次形式
$$\Delta(\bm{v})=\bm{v}^T Q \bm{v}$$
で表せる。

## ライブラリ
```
numpy
torch
```

## デモ

```
python simplification.py
```

<table>
  <tr>
    <td width="24%">入力</td>
    <td width="24%">簡略化(50%)</td>
    <td width="24%">簡略化(20%)</td>
    <td width="24%">簡略化(1%)</td>
  </tr>
  <tr>
    <td width="24%"><img src="docs/original.png" width="100%"/></td>
    <td width="24%"><img src="docs/simp_v1.png" width="100%"/></td>
    <td width="24%"><img src="docs/simp_v2.png" width="100%"/></td>
    <td width="24%"><img src="docs/simp_v4.png" width="100%"/></td>
  </tr>

  <tr>
    <td width="24%">14762 vertices</td>
    <td width="24%">7381 vertices</td>
    <td width="24%">2952 vertices</td>
    <td width="24%">147 vertices</td>
  </tr>
  <tr>
    <td width="24%">29520 faces</td>
    <td width="24%">14758 faces</td>
    <td width="24%">5900 faces</td>
    <td width="24%">290 faces</td>
  </tr>
</table>

本スクリプトは改良途中。
非多様体を生じるエッジ縮約や、境界エッジの縮約は行わない。
エッジ角度を考慮していないため、自己交差や面のフリップが生じうる。

## TODO

- （縮約すると多様体を生じる）valance=3の頂点の削除
- エッジ角度を考慮した縮約
- 更新後頂点位置の最適化（現在は中点）