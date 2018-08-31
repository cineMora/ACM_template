//1.链式前向星
void ins(int u, int v, int w)
{
    cnt++;
    e[cnt].to = v;
    e[cnt].next = head[u];
    e[cnt].w = w;
    head[u] = cnt;
}
for (int i = head[x]; i; i = e[i].next)
{
    v = e[i].to;
    u = x;
    w = e[i].w;
}
//2.并查集（不要忘记初始化）
//①普通并查集
int findset(int x) { reuturn pa[x] != x ? pa[x] = findset(pa[x]) : x; }
//②使用补集思想（团伙、食物链）
x = e[i].a;
y = e[i].b;
v = e[i].c;
x = findf(x);
y = findf(y);
if (x == y)
{
    cout << v;
    return 0;
}
fa[x] = findf(e[i].b + n);
fa[y] = findf(e[i].a + n);
//③ 带权并查集 （银河英雄传说）
int findf(int x)
{
    if (x != f[x])
    {
        int r = findf(f[x]);
        d[x] += d[f[x]];
        f[x] = r;
    }
    return f[x];
}
for (int i = 1; i <= 30000; i++)
{
    f[i] = i;
    sz[i] = 1;
}
if (s == 'M')
{
    scanf("%d%d", &u, &v);
    u = findf(u);
    v = findf(v);
    f[u] = v;
    d[u] = sz[v];
    sz[v] += sz[u];
}
else if (s == 'C')
{
    scanf("%d%d", &u, &v);
    ur = findf(u);
    vr = findf(v);
    if (ur == vr)
        printf("%d\n", abs(d[u] - d[v]) - 1);
    else
        printf("-1\n");
}
//3.最短路
//①floyd
void floyd()
{
    for (int k = 1; k <= n; k++)
    {
        for (int i = 1; i <= n; i++)
        {
            if (k == i)
                continue;
            for (int j = 1; j <= n; j++)
            {
                if (k != i && k != j && i != j)
                    g[i][j] = min(g[i][j], g[i][k] + g[k][j]);
            }
        }
    }
}
//②spfa
void spfa(int u)
{
    int i, v, w;
    for (i = 1; i <= n; i++)
    {
        dist[i] = inf;
        inq[i] = false;
    }
    queue<int> q;
    q.push(u);
    inq[u] = true;
    dist[u] = 0;
    while (!q.empty())
    {
        u = q.front();
        q.pop();
        inq[u] = false;
        for (i = 0; i < g[u].size(); i++)
        {
            v = g[u][i].v;
            w = g[u][i].w;
            if (dist[u] + w < dist[v])
            {
                dist[v] = dist[u] + w;
                if (!inq[v])
                {
                    q.push(v);
                    inq[v] = true;
                }
            }
        }
    }
}
//③普通dij(n^2)
int Dijkstra(int s, int e)
{
    d[s] = 0;

    for (int v = -1; 1; v = -1)
    {
        for (int i = 1; i <= V; i++)
            if (!used[i] && (v == -1 || d[i] < d[v]))
                v = i;

        if (v == -1)
            break;

        used[v] = 1;

        for (int i = 1; i <= V; i++)
            d[i] = min(d[i], d[v] + graph[v][i]);
    }

    return d[e];
}
//④堆优化dij（sth版）
struct orz
{
    int d, p;
    friend bool operator<(orz a, orz b) { return a.d > b.d; } //堆和set里面都只有小于号，所以要用小根堆的话要将<重定向为>
};
struct Edge
{
    int to;
    int w;
};
priority_queue<orz> ss;
void dij(int s)
{

    d[s] = 0;
    orz tmp;
    tmp.d = 0, tmp.p = s;
    ss.push(tmp);
    flag++;
    int x, dd;
    Edge j;
    while (!ss.empty()) //不能只做n次，要一直做到堆空
    {
        tmp = ss.top();
        ss.pop();
        x = tmp.p, dd = tmp.d;
        if (v[x] == flag)
            continue; //这里一定要判断！！！
        v[x] = flag;
        for (int i = 0; i < edge[x].size(); i++)
        {

            j = edge[x][i];
            if (d[j.to] > dd + j.w)
            {
                d[j.to] = dd + j.w;
                tmp.d = dd + j.w, tmp.p = j.to;
                ss.push(tmp);
            }
        }
    }
}
//⑤求最短路径条数（社交网络）
for (int i = 1; i <= m; i++)
{
    scanf("%d%d%d", &u, &v, &w);
    a[u][v] = a[v][u] = w;
    cnts[u][v] = cnts[v][u] = 1;
}
for (int k = 1; k <= n; k++)
{
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            if (k != i && k != j && i != j)
            {
                if (a[i][j] > a[i][k] + a[k][j])
                {
                    a[i][j] = a[i][k] + a[k][j];
                    cnts[i][j] = cnts[i][k] * cnts[k][j];
                }
                else if (a[i][j] == a[i][k] + a[k][j])
                {
                    cnts[i][j] += cnts[i][k] * cnts[k][j];
                }
            }
        }
    }
}
//4.最小生成树
//poj 1287
#include <iostream>
#include <cstdio>
#include <cmath>
#include <string>
#include <cstring>
#include <algorithm>
#include <vector>
#include <queue>
#define ll long long
#define mx 100005
using namespace std;
struct edge
{
    int u;
    int v;
    long long w;
};
int n, k, m, f[mx], vis[mx], pre[mx], fa, fb;
ll ans;
edge g[mx], tmp;
int findf(int x)
{
    return x == f[x] ? x : f[x] = findf(f[x]);
}
bool cmp(edge a, edge b)
{
    return a.w < b.w;
}
int main()
{
    while (1)
    {
        ans = 0;
        cin >> n;
        if (n == 0)
            break;
        cin >> m;
        long long u, v, w, ans = 0, sum = 0;
        for (int i = 1; i <= n; i++)
        {
            f[i] = i;
        }
        for (int i = 1; i <= m; i++)
        {
            cin >> u >> v >> w;
            tmp.u = u;
            tmp.v = v;
            tmp.w = w;
            g[i] = tmp;
        }
        sort(g + 1, g + 1 + m, cmp);

        for (int i = 1; i <= m; i++)
        {
            fa = findf(g[i].u);
            fb = findf(g[i].v);
            if (fa == fb)
                continue;
            f[fa] = fb;
            ans += g[i].w;
        }
        cout << ans << endl;
    }
    return 0;
}
//5.欧拉回路
//hdu 1878
//是否存在欧拉回路
#include <iostream>
#include <cstdio>
#include <string>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>
#define ll long long
#define fo(i, l, r) for (int i = l; i <= r; i++)
#define fd(i, l, r) for (int i = r; i >= l; i--)
using namespace std;
const int maxn = 1050;
ll read()
{
    ll x = 0, f = 1;
    char ch = getchar();
    while (!(ch >= '0' && ch <= '9'))
    {
        if (ch == '-')
            f = -1;
        ch = getchar();
    };
    while (ch >= '0' && ch <= '9')
    {
        x = x * 10 + (ch - '0');
        ch = getchar();
    };
    return x * f;
}
int n, m, u, v, d[maxn];
bool vis[maxn];
vector<int> g[maxn];
void dfs(int x)
{
    vis[x] = true;
    for (int i = 0; i < g[x].size(); i++)
    {
        if (!vis[g[x][i]])
        {
            dfs(g[x][i]);
        }
    }
    return;
}
int main()
{
    while (1)
    {
        n = read();
        if (n == 0)
            break;
        m = read();
        memset(d, 0, sizeof(d));
        fo(i, 1, n) g[i].clear();
        fo(i, 1, m)
        {
            u = read();
            v = read();
            d[u]++;
            d[v]++;
            g[u].push_back(v);
            g[v].push_back(u);
        }
        memset(vis, 0, sizeof(vis));
        dfs(1);
        bool ok = true;
        fo(i, 1, n)
        {
            if (!vis[i])
            {
                cout << 0 << endl;
                ok = false;
                break;
            }
        }
        if (!ok)
            continue;

        fo(i, 1, n)
        {
            if (d[i] % 2 != 0)
            {
                cout << 0 << endl;
                ok = false;
                break;
            }
        }
        if (ok)
            cout << 1 << endl;
    }
    return 0;
}
//HRBUST - 2310
//存在多少条欧拉路径？
#include <iostream>
#include <cstdio>
#include <string>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>
#define ll long long
#define fo(i, l, r) for (int i = l; i <= r; i++)
#define fd(i, l, r) for (int i = r; i >= l; i--)
using namespace std;
const int maxn = 100500;
ll read()
{
    ll x = 0, f = 1;
    char ch = getchar();
    while (!(ch >= '0' && ch <= '9'))
    {
        if (ch == '-')
            f = -1;
        ch = getchar();
    };
    while (ch >= '0' && ch <= '9')
    {
        x = x * 10 + (ch - '0');
        ch = getchar();
    };
    return x * f;
}
int n, m, u, v, d[maxn], f[maxn];
int main()
{
    int T;
    cin >> T;
    while (T--)
    {
        memset(f, 0, sizeof(f));
        memset(d, 0, sizeof(d));
        n = read();
        fo(i, 1, n - 1)
        {
            u = read();
            v = read();
            d[u]++;
            d[v]++;
        }
        int ans = 0;
        fo(i, 1, n)
        {
            if (d[i] % 2 == 1)
                ans++;
        }
        cout << ans / 2 << endl;
    }
    return 0;
}
//6.差分约束
//HYSBZ - 2330 糖果
/*
如果X=1， 表示第A个小朋友分到的糖果必须和第B个小朋友分到的糖果一样多；
如果X=2， 表示第A个小朋友分到的糖果必须少于第B个小朋友分到的糖果；
如果X=3， 表示第A个小朋友分到的糖果必须不少于第B个小朋友分到的糖果；
如果X=4， 表示第A个小朋友分到的糖果必须多于第B个小朋友分到的糖果；
如果X=5， 表示第A个小朋友分到的糖果必须不多于第B个小朋友分到的糖果；
*/
#include <iostream>
#include <cstdio>
#include <string>
#include <cstring>
#include <algorithm>
#include <queue>
#define ll long long
using namespace std;
const int maxn = 100050;
struct edge
{
    int v;
    int w;
    int nxt;
} e[maxn * 3];
int n, k;
int cnt, head[maxn];
int rd[maxn];
bool vis[maxn];
ll dis[maxn];
int read()
{
    char ch = getchar();
    int f = 1, x = 0;
    while (!(ch >= '0' && ch <= '9'))
    {
        if (ch == '-')
            f = -1;
        ch = getchar();
    };
    while (ch >= '0' && ch <= '9')
    {
        x = x * 10 + (ch - '0');
        ch = getchar();
    };
    return x * f;
}
int ins(int u, int v, int w)
{
    cnt++;
    e[cnt].v = v;
    e[cnt].w = w;
    e[cnt].nxt = head[u];
    head[u] = cnt;
}
bool spfa()
{
    int now, nt;
    queue<int> q;
    for (int i = 1; i <= n; i++)
    {
        q.push(i);
        dis[i] = rd[i] = 1;
    }
    while (!q.empty())
    {
        now = q.front();
        q.pop();
        for (int i = head[now]; i; i = e[i].nxt)
        {
            nt = e[i].v;
            if (dis[nt] < dis[now] + e[i].w)
            {
                dis[nt] = dis[now] + e[i].w;
                if (!vis[nt])
                {
                    vis[nt] = true;
                    q.push(nt);
                    rd[nt]++;
                    if (rd[nt] > n)
                        return false;
                }
            }
        }
        vis[now] = false;
    }
    return true;
}
int main()
{
    n = read();
    k = read();
    int x, a, b;
    for (int i = 1; i <= k; i++)
    {
        x = read();
        a = read();
        b = read();
        if (x == 1)
        {
            ins(a, b, 0);
            ins(b, a, 0);
        }
        else if (x == 2)
        {
            if (a == b)
            {
                cout << -1;
                return 0;
            }
            ins(a, b, 1);
        }
        else if (x == 3)
        {
            ins(b, a, 0);
        }
        else if (x == 4)
        {
            if (a == b)
            {
                cout << -1;
                return 0;
            }
            ins(b, a, 1);
        }
        else
        {
            ins(a, b, 0);
        }
    }
    ll ans = 0;
    if (spfa())
    {
        for (int i = 1; i <= n; i++)
            ans += dis[i];
        cout << ans;
    }
    else
    {
        cout << -1;
    }
    return 0;
}
//7.拓扑排序
//①dfs（lrj版）
vector<int> G[mx];            //邻接表存储图
int c[mx], topo[mx], t, n, m; //c表示是否访问过（1为访问过，0为未访问，-1为正访问），topo表示拓扑序，t为倒序的排位指针，n为节点数，m为边数
bool dfs(int u)
{
    c[u] = -1; // 访问标志
    for (int i = 0, v; i < G[u].size(); i++)
    {
        v = G[u][i]; //遍历出边
        if (c[v] < 0)
            return false; //存在有向环，失败退出
        else if (!c[v] && !dfs(v))
            return false; //如果已经访问，或者自身往下形不成拓扑序，失败退出；
    }
    c[u] = 1;
    topo[--t] = u;
    return true;
}
bool topsort()
{
    t = n;
    memset(c, 0, sizeof(c));
    for (int u = 0; u < n; u++)
        if (!c[u])
            if (!dfs(u))
                return false;
    return true;
}
void output()
{
    if (topsort())
    {
        for (int i = 0; i < n; i++)
            cout << topo[i] << " "; //输出拓扑序列
        cout << endl;
    }
    else
    {
        cout << "No way!" << endl;
    }
}
//②有向无环图上的最长链
void dfs(int x)
{
    if (!g[x].size())
    {
        f[x] = 1;
        return;
    }
    for (int i = 0; i < g[x].size(); i++)
    {
        if (!f[g[x][i]])
            dfs(g[x][i]);
        f[x] = (f[x] + f[g[x][i]]) % mod;
    }
}
void dp()
{
    for (int i = 1; i <= n; i++)
    {
        if (!f[topo[i]])
            dfs(topo[i]);
        if (!ind[topo[i]])
            ans = (ans + f[topo[i]]) % mod;
    }
    cout << ans;
}
//8.二分图匹配
//①一般匹配
/* *******************************
* 二分图匹配（Hopcroft-Karp 算法）
* 复杂度 O(sqrt(n)*E)
* 邻接表存图， vector 实现
* vector 先初始化，然后假如边
* uN 为左端的顶点数，使用前赋值 (点编号 0 开始)
*/
const int MAXN = 3000;
const int INF = 0x3f3f3f3f;
vector<int> G[MAXN];
int uN;
int Mx[MAXN], My[MAXN];
int dx[MAXN], dy[MAXN];
int dis;
bool used[MAXN];
bool SearchP()
{
    queue<int> Q;
    dis = INF;
    memset(dx, -1, sizeof(dx));
    memset(dy, -1, sizeof(dy));
    for (int i = 0; i < uN; i++)
        if (Mx[i] == -1)
        {
            Q.push(i);
            dx[i] = 0;
        }
    while (!Q.empty())
    {
        int u = Q.front();
        Q.pop();
        if (dx[u] > dis)
            break;
        int sz = G[u].size();
        for (int i = 0; i < sz; i++)
        {
            int v = G[u][i];
            if (dy[v] == -1)
            {
                dy[v] = dx[u] + 1;
                if (My[v] == -1)
                    dis = dy[v];
                else
                {
                    dx[My[v]] = dy[v] + 1;
                    Q.push(My[v]);
                }
            }
        }
    }
    return dis != INF;
}
bool DFS(int u)
{
    int sz = G[u].size();
    for (int i = 0; i < sz; i++)
    {
        int v = G[u][i];
        if (!used[v] && dy[v] == dx[u] + 1)
        {
            used[v] = true;
            if (My[v] != -1 && dy[v] == dis)
                continue;
            if (My[v] == -1 || DFS(My[v]))
            {
                My[v] = u;
                Mx[u] = v;
                return true;
            }
        }
    }
    return false;
}
int MaxMatch()
{
    int res = 0;
    memset(Mx, -1, sizeof(Mx));
    memset(My, -1, sizeof(My));
    while (SearchP())
    {
        memset(used, false, sizeof(used));
        for (int i = 0; i < uN; i++)
            if (Mx[i] == -1 && DFS(i))
                res++;
    }
    return res;
}
/*
* 匈牙利算法邻接表形式
* 使用前用 init() 进行初始化，给 uN 赋值
* 加边使用函数 addedge(u,v)
*
*/
const int MAXN = 5010;  //点数的最大值
const int MAXM = 50010; //边数的最大值
struct Edge
{
    int to, next;
} edge[MAXM];
int head[MAXN], tot;
void init()
{
    tot = 0;
    memset(head, -1, sizeof(head));
}
void addedge(int u, int v)
{
    edge[tot].to = v;
    edge[tot].next = head[u];
    head[u] = tot++;
}
int linker[MAXN];
bool used[MAXN];
int uN;
bool dfs(int u)
{
    for (int i = head[u]; i != -1; i = edge[i].next)
    {
        int v = edge[i].to;
        if (!used[v])
        {
            used[v] = true;
            if (linker[v] == -1 || dfs(linker[v]))
            {
                linker[v] = u;
                return true;
            }
        }
    }
    return false;
}
int hungary()
{
    int res = 0;
    memset(linker, -1, sizeof(linker));
    //点的编号 0∼uN-1
    for (int u = 0; u < uN; u++)
    {
        memset(used, false, sizeof(used));
        if (dfs(u))
            res++;
    }
    return res;
}
/* ***********************************************************
//二分图匹配（匈牙利算法的 DFS 实现） (邻接矩阵形式)
//初始化： g[][] 两边顶点的划分情况
//建立 g[i][j] 表示 i->j 的有向边就可以了，是左边向右边的匹配
//g 没有边相连则初始化为 0
//uN 是匹配左边的顶点数， vN 是匹配右边的顶点数
//调用： res=hungary(); 输出最大匹配数
//优点：适用于稠密图， DFS 找增广路，实现简洁易于理解
//时间复杂度:O(VE)
//*************************************************************/
//顶点编号从 0 开始的
const int MAXN = 510;
int uN, vN;        //u,v 的数目，使用前面必须赋值
int g[MAXN][MAXN]; //邻接矩阵
int linker[MAXN];
bool used[MAXN];
bool dfs(int u)
{
    for (int v = 0; v < vN; v++)
        if (g[u][v] && !used[v])
        {
            used[v] = true;
            if (linker[v] == -1 || dfs(linker[v]))
            {
                linker[v] = u;
                return true;
            }
        }
    return false;
}
int hungary()
{
    int res = 0;
    memset(linker, -1, sizeof(linker));
    for (int u = 0; u < uN; u++)
    {
        memset(used, false, sizeof(used));
        if (dfs(u))
            res++;
    }
    return res;
}
//②多重匹配
const int MAXN = 1010;
const int MAXM = 510;
int uN, vN;
int g[MAXN][MAXM];
int linker[MAXM][MAXN];
bool used[MAXM];
int num[MAXM]; //右边最大的匹配数
bool dfs(int u)
{
    for (int v = 0; v < vN; v++)
        if (g[u][v] && !used[v])
        {
            used[v] = true;
            if (linker[v][0] < num[v])
            {
                linker[v][++linker[v][0]] = u;
                return true;
            }
            for (int i = 1; i <= num[v]; i++)
                if (dfs(linker[v][i]))
                {
                    linker[v][i] = u;
                    return true;
                }
        }
    return false;
}
int hungary()
{
    int res = 0;
    for (int i = 0; i < vN; i++)
        linker[i][0] = 0;
    for (int u = 0; u < uN; u++)
    {
        memset(used, false, sizeof(used));
        if (dfs(u))
            res++;
    }
    return res;
}
//③二分图最大权匹配
/* KM 算法
* 复杂度 O(nx*nx*ny)
* 求最大权匹配
* 若求最小权匹配，可将权值取相反数，结果取相反数
* 点的编号从 0 开始
*/
const int N = 310;
const int INF = 0x3f3f3f3f;
int nx, ny;                  //两边的点数
int g[N][N];                 //二分图描述
int linker[N], lx[N], ly[N]; //y 中各点匹配状态， x,y 中的点标号
int slack[N];
bool visx[N], visy[N];
bool DFS(int x)
{
    visx[x] = true;
    for (int y = 0; y < ny; y++)
    {
        if (visy[y])
            continue;
        int tmp = lx[x] + ly[y] - g[x][y];
        if (tmp == 0)
        {
            visy[y] = true;
            if (linker[y] == -1 || DFS(linker[y]))
            {
                linker[y] = x;
                return true;
            }
        }
        else if (slack[y] > tmp)
            slack[y] = tmp;
    }
    return false;
}
int KM()
{
    memset(linker, -1, sizeof(linker));
    memset(ly, 0, sizeof(ly));
    for (int i = 0; i < nx; i++)
    {
        lx[i] = -INF;
        for (int j = 0; j < ny; j++)
            if (g[i][j] > lx[i])
                lx[i] = g[i][j];
    }
    for (int x = 0; x < nx; x++)
    {
        for (int i = 0; i < ny; i++)
            slack[i] = INF;
        while (true)
        {
            memset(visx, false, sizeof(visx));
            memset(visy, false, sizeof(visy));
            if (DFS(x))
                break;
            int d = INF;
            for (int i = 0; i < ny; i++)
                if (!visy[i] && d > slack[i])
                    d = slack[i];
            for (int i = 0; i < nx; i++)
                if (visx[i])
                    lx[i] -= d;
            for (int i = 0; i < ny; i++)
            {
                if (visy[i])
                    ly[i] += d;
                else
                    slack[i] -= d;
            }
        }
    }
    int res = 0;
    for (int i = 0; i < ny; i++)
        if (linker[i] != -1)
            res += g[linker[i]][i];
    return res;
}
//④一般图匹配
//HDU 2255
int main()
{
    int n;
    while (scanf("%d", &n) == 1)
    {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                scanf("%d", &g[i][j]);
        nx = ny = n;
        printf("%d\n", KM());
    }
    return 0;
}
const int MAXN = 250;
int N; //点的个数，点的编号从 1 到 N
bool Graph[MAXN][MAXN];
int Match[MAXN];
bool InQueue[MAXN], InPath[MAXN], InBlossom[MAXN];
int Head, Tail;
int Queue[MAXN];
int Start, Finish;
int NewBase;
int Father[MAXN], Base[MAXN];
int Count; //匹配数，匹配对数是 Count/2
void CreateGraph()
{
    int u, v;
    memset(Graph, false, sizeof(Graph));
    scanf("%d", &N);
    while (scanf("%d%d", &u, &v) == 2)
    {
        Graph[u][v] = Graph[v][u] = true;
    }
}
void Push(int u)
{
    Queue[Tail] = u;
    Tail++;
    InQueue[u] = true;
}
int Pop()
{
    int res = Queue[Head];
    Head++;
    return res;
}
int FindCommonAncestor(int u, int v)
{
    memset(InPath, false, sizeof(InPath));
    while (true)
    {
        u = Base[u];
        InPath[u] = true;
        if (u == Start)
            break;
        u = Father[Match[u]];
    }
    while (true)
    {
        v = Base[v];
        if (InPath[v])
            break;
        v = Father[Match[v]];
    }
    return v;
}
void ResetTrace(int u)
{
    int v;
    while (Base[u] != NewBase)
    {
        v = Match[u];
        InBlossom[Base[u]] = InBlossom[Base[v]] = true;
        u = Father[v];
        if (Base[u] != NewBase)
            Father[u] = v;
    }
}
void BloosomContract(int u, int v)
{
    NewBase = FindCommonAncestor(u, v);
    memset(InBlossom, false, sizeof(InBlossom));
    ResetTrace(u);
    ResetTrace(v);
    if (Base[u] != NewBase)
        Father[u] = v;
    if (Base[v] != NewBase)
        Father[v] = u;
    for (int tu = 1; tu <= N; tu++)
        if (InBlossom[Base[tu]])
        {
            Base[tu] = NewBase;
            if (!InQueue[tu])
                Push(tu);
        }
}
void FindAugmentingPath()
{
    memset(InQueue, false, sizeof(InQueue));
    memset(Father, 0, sizeof(Father));
    for (int i = 1; i <= N; i++)
        Base[i] = i;
    Head = Tail = 1;
    Push(Start);
    Finish = 0;
    while (Head < Tail)
    {
        int u = Pop();
        for (int v = 1; v <= N; v++)
            if (Graph[u][v] && (Base[u] != Base[v]) && (Match[u] != v))
            {
                if ((v == Start) || ((Match[v] > 0) && Father[Match[v]] > 0))
                    BloosomContract(u, v);
                else if (Father[v] == 0)
                {
                    Father[v] = u;
                    if (Match[v] > 0)
                        Push(Match[v]);
                    else
                    {
                        Finish = v;
                        return;
                    }
                }
            }
    }
}
void AugmentPath()
{
    int u, v, w;
    u = Finish;
    while (u > 0)
    {
        v = Father[u];
        w = Match[v];
        Match[v] = u;
        Match[u] = v;
        u = w;
    }
}
void Edmonds()
{
    memset(Match, 0, sizeof(Match));
    for (int u = 1; u <= N; u++)
        if (Match[u] == 0)
        {
            Start = u;
            FindAugmentingPath();
            if (Finish > 0)
                AugmentPath();
        }
}
void PrintMatch()
{
    Count = 0;
    for (int u = 1; u <= N; u++)
        if (Match[u] > 0)
            Count++;
    printf("%d\n", Count);
    for (int u = 1; u <= N; u++)
        if (u < Match[u])
            printf("%d␣%d\n", u, Match[u]);
}
int main()
{
    CreateGraph(); //建图
    Edmonds();     //进行匹配
    PrintMatch();  //输出匹配数和匹配
    return 0;
}
//⑤一般图的最大加权匹配模板
//注意 G 的初始化, 需要偶数个点，刚好可以形成 n/2 个匹配
//如果要求最小权匹配，可以取相反数，或者稍加修改就可以了
//点的编号从 0 开始的
const int MAXN = 110;
const int INF = 0x3f3f3f3f;
int G[MAXN][MAXN];
int cnt_node; //点的个数
int dis[MAXN];
int match[MAXN];
bool vis[MAXN];
int sta[MAXN], top; //堆栈
bool dfs(int u)
{
    sta[top++] = u;
    if (vis[u])
        return true;
    vis[u] = true;
    for (int i = 0; i < cnt_node; i++)
        if (i != u && i != match[u] && !vis[i])
        {
            int t = match[i];
            if (dis[t] < dis[u] + G[u][i] - G[i][t])
            {
                dis[t] = dis[u] + G[u][i] - G[i][t];
                if (dfs(t))
                    return true;
            }
        }
    top--;
    vis[u] = false;
    return false;
}
int P[MAXN];
//返回最大匹配权值
int get_Match(int N)
{
    cnt_node = N;
    for (int i = 0; i < cnt_node; i++)
        P[i] = i;
    for (int i = 0; i < cnt_node; i += 2)
    {
        match[i] = i + 1;
        match[i + 1] = i;
    }
    int cnt = 0;
    while (1)
    {
        memset(dis, 0, sizeof(dis));
        memset(vis, false, sizeof(vis));
        top = 0;
        bool update = false;
        for (int i = 0; i < cnt_node; i++)
            if (dfs(P[i]))
            {
                update = true;
                int tmp = match[sta[top - 1]];
                int j = top - 2;
                while (sta[j] != sta[top - 1])
                {
                    match[tmp] = sta[j];
                    swap(tmp, match[sta[j]]);
                    j--;
                }
                match[tmp] = sta[j];
                match[sta[j]] = tmp;
                break;
            }
        if (!update)
        {
            cnt++;
            if (cnt >= 3)
                break;
            random_shuffle(P, P + cnt_node);
        }
    }
    int ans = 0;
    for (int i = 0; i < cnt_node; i++)
    {
        int v = match[i];
        if (i < v)
            ans += G[i][v];
    }
    return ans;
}
//9.网络流
//①最大流ISAP
#include <iostream>
#include <cstdio>
#include <string>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>
#include <queue>
#include <stack>
#define ll long long
#define fo(i, l, r) for (int i = l; i <= r; i++)
#define fd(i, l, r) for (int i = r; i >= l; i--)
using namespace std;
ll read()
{
    ll x = 0, f = 1;
    char ch = getchar();
    while (!(ch >= '0' && ch <= '9'))
    {
        if (ch == '-')
            f = -1;
        ch = getchar();
    };
    while (ch >= '0' && ch <= '9')
    {
        x = x * 10 + (ch - '0');
        ch = getchar();
    };
    return x * f;
}
const int MAXN = 100010; //点数的最大值
const int MAXM = 400010; //边数的最大值
const int INF = 0x3f3f3f3f;
struct Edge
{
    int to, next, cap, flow;
} edge[MAXM]; //注意是 MAXM
int tol;
int head[MAXN];
int gap[MAXN], dep[MAXN], cur[MAXN];
void init()
{
    tol = 0;
    memset(head, -1, sizeof(head));
}
void addedge(int u, int v, int w, int rw = 0)
{
    edge[tol].to = v;
    edge[tol].cap = w;
    edge[tol].flow = 0;
    edge[tol].next = head[u];
    head[u] = tol++;
    edge[tol].to = u;
    edge[tol].cap = rw;
    edge[tol].flow = 0;
    edge[tol].next = head[v];
    head[v] = tol++;
}
int Q[MAXN];
void BFS(int start, int end)
{
    memset(dep, -1, sizeof(dep));
    memset(gap, 0, sizeof(gap));
    gap[0] = 1;
    int front = 0, rear = 0;
    dep[end] = 0;
    Q[rear++] = end;
    while (front != rear)
    {
        int u = Q[front++];

        for (int i = head[u]; i != -1; i = edge[i].next)
        {
            int v = edge[i].to;
            if (dep[v] != -1)
                continue;
            Q[rear++] = v;
            dep[v] = dep[u] + 1;
            gap[dep[v]]++;
        }
    }
}
int S[MAXN];
int sap(int start, int end, int N)
{
    BFS(start, end);
    memcpy(cur, head, sizeof(head));
    int top = 0;
    int u = start;
    int ans = 0;
    while (dep[start] < N)
    {
        if (u == end)
        {
            int Min = INF;
            int inser;
            for (int i = 0; i < top; i++)
                if (Min > edge[S[i]].cap - edge[S[i]].flow)
                {
                    Min = edge[S[i]].cap - edge[S[i]].flow;
                    inser = i;
                }
            for (int i = 0; i < top; i++)
            {
                edge[S[i]].flow += Min;
                edge[S[i] ^ 1].flow -= Min;
            }
            ans += Min;
            top = inser;
            u = edge[S[top] ^ 1].to;
            continue;
        }
        bool flag = false;
        int v;
        for (int i = cur[u]; i != -1; i = edge[i].next)
        {
            v = edge[i].to;
            if (edge[i].cap - edge[i].flow && dep[v] + 1 == dep[u])
            {
                flag = true;
                cur[u] = i;
                break;
            }
        }
        if (flag)
        {
            S[top++] = cur[u];
            u = v;
            continue;
        }
        int Min = N;
        for (int i = head[u]; i != -1; i = edge[i].next)
            if (edge[i].cap - edge[i].flow && dep[edge[i].to] < Min)
            {
                Min = dep[edge[i].to];
                cur[u] = i;
            }
        gap[dep[u]]--;
        if (!gap[dep[u]])
            return ans;
        dep[u] = Min + 1;
        gap[dep[u]]++;
        if (u != start)
            u = edge[S[--top] ^ 1].to;
    }
    return ans;
}
int main()
{
    int n, m;
    while (scanf("%d%d", &m, &n) == 2)
    {

        memset(gap, 0, sizeof(gap));
        memset(cur, 0, sizeof(cur));
        memset(dep, 0, sizeof(dep));
        //	m = read();
        //	n = read();
        int u, v, w;
        init();
        fo(i, 1, m)
        {
            u = read();
            v = read();
            u--;
            v--;
            w = read();
            addedge(u, v, w);
        }
        cout << sap(0, n - 1, n) << endl;
    }
    return 0;
}
//②判断多解
//判断最大流多解就是在残留网络中找正环
bool vis[MAXN], no[MAXN];
int Stack[MAXN], top;
bool dfs(int u, int pre, bool flag)
{
    vis[u] = 1;
    Stack[top++] = u;
    for (int i = head[u]; i != -1; i = edge[i].next)
    {
        int v = edge[i].to;
        if (edge[i].cap <= edge[i].flow)
            continue;
        if (v == pre)
            continue;
        if (!vis[v])
        {
            if (dfs(v, u, edge[i ^ 1].flow < edge[i ^ 1].cap))
                return true;
        }
        else if (!no[v])
            return true;
    }
    if (!flag)
    {
        while (1)
        {
            int v = Stack[--top];
            no[v] = true;
            if (v == u)
                break;
        }
    }
    return false;
}
//执行完最大流后可进行调用
memset(vis, false, sizeof(vis));
memset(no, false, sizeof(no));
top = 0;
bool flag = dfs(end, end, 0);
//③费用流
//最小费用最大流，求最大费用只需要取相反数，结果取相反数即可。
//点的总数为 N，点的编号 0 ∼ N-1
const int MAXN = 10000;
const int MAXM = 100000;
const int INF = 0x3f3f3f3f;
struct Edge
{
    int to, next, cap, flow, cost;
} edge[MAXM];
int head[MAXN], tol;
int pre[MAXN], dis[MAXN];
bool vis[MAXN];
int N; //节点总个数，节点编号从 0∼N-1
void init(int n)
{
    N = n;
    tol = 0;
    memset(head, -1, sizeof(head));
}
void addedge(int u, int v, int cap, int cost)
{
    edge[tol].to = v;
    edge[tol].cap = cap;
    edge[tol].cost = cost;
    edge[tol].flow = 0;
    edge[tol].next = head[u];
    head[u] = tol++;
    edge[tol].to = u;
    edge[tol].cap = 0;
    edge[tol].cost = -cost;
    edge[tol].flow = 0;
    edge[tol].next = head[v];
    head[v] = tol++;
}
bool spfa(int s, int t)
{
    queue<int> q;
    for (int i = 0; i < N; i++)
    {
        dis[i] = INF;
        vis[i] = false;
        pre[i] = -1;
    }
    dis[s] = 0;
    vis[s] = true;
    q.push(s);
    while (!q.empty())
    {
        int u = q.front();
        q.pop();
        vis[u] = false;
        for (int i = head[u]; i != -1; i = edge[i].next)
        {
            int v = edge[i].to;
            if (edge[i].cap > edge[i].flow && dis[v] > dis[u] + edge
                                                                    [i]
                                                                        .cost)
            {
                dis[v] = dis[u] + edge[i].cost;
                pre[v] = i;
                if (!vis[v])
                {
                    vis[v] = true;
                    q.push(v);
                }
            }
        }
    }
    if (pre[t] == -1)
        return false;
    else
        return true;
}
//返回的是最大流， cost 存的是最小费用
int minCostMaxflow(int s, int t, int &cost)
{
    int flow = 0;
    cost = 0;
    while (spfa(s, t))
    {
        int Min = INF;
        for (int i = pre[t]; i != -1; i = pre[edge[i ^ 1].to])
        {
            if (Min > edge[i].cap - edge[i].flow)
                Min = edge[i].cap - edge[i].flow;
        }
        for (int i = pre[t]; i != -1; i = pre[edge[i ^ 1].to])
        {
            edge[i].flow += Min;
            edge[i ^ 1].flow -= Min;
            cost += edge[i].cost * Min;
        }
        flow += Min;
    }
    return flow;
}
//例题：网络扩容
#include <iostream>
#include <cstdio>
#include <string>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>
#include <queue>
#include <stack>
#define ll long long
#define fo(i, l, r) for (int i = l; i <= r; i++)
#define fd(i, l, r) for (int i = r; i >= l; i--)
using namespace std;
ll read()
{
    ll x = 0, f = 1;
    char ch = getchar();
    while (!(ch >= '0' && ch <= '9'))
    {
        if (ch == '-')
            f = -1;
        ch = getchar();
    };
    while (ch >= '0' && ch <= '9')
    {
        x = x * 10 + (ch - '0');
        ch = getchar();
    };
    return x * f;
}
const int MAXN = 10100; //??????
const int MAXM = 40010; //??????
const ll INF = 0x3f3f3f3f3f3f3f3f;
struct Edge
{
    int u, to, next;
    ll cap, flow, cost, caonima;
} edge[MAXM]; //??? MAXM
int tol;
ll head[MAXN];
ll gap[MAXN], dep[MAXN];
ll cur[MAXN];
ll pre[MAXN], dis[MAXN];
bool vis[MAXN];
int N;
void init(int n)
{
    N = n;
    tol = 0;
    memset(head, -1, sizeof(head));
}
void addedge(int u, int v, ll w, ll cost, ll rw = 0)
{
    edge[tol].u = u;
    edge[tol].to = v;
    edge[tol].cap = w;
    edge[tol].caonima = cost;
    edge[tol].flow = 0;
    edge[tol].next = head[u];
    edge[tol].cost = 0;
    head[u] = tol++;
    edge[tol].u = v;
    edge[tol].to = u;
    edge[tol].cap = rw;
    edge[tol].caonima = -cost;
    edge[tol].flow = 0;
    edge[tol].next = head[v];
    edge[tol].cost = 0;
    head[v] = tol++;
}
void addedge2(int u, int v, ll w, ll cost, ll rw = 0)
{
    edge[tol].u = u;
    edge[tol].to = v;
    edge[tol].cap = w;
    edge[tol].cost = cost;
    edge[tol].flow = 0;
    edge[tol].next = head[u];
    head[u] = tol++;
    edge[tol].u = v;
    edge[tol].to = u;
    edge[tol].cap = rw;
    edge[tol].cost = -cost;
    edge[tol].flow = 0;
    edge[tol].next = head[v];
    head[v] = tol++;
}
int Q[MAXN];
void BFS(int start, int end)
{
    memset(dep, -1, sizeof(dep));
    memset(gap, 0, sizeof(gap));
    gap[0] = 1;
    int front = 0, rear = 0;
    dep[end] = 0;
    Q[rear++] = end;
    while (front != rear)
    {
        int u = Q[front++];

        for (int i = head[u]; i != -1; i = edge[i].next)
        {
            int v = edge[i].to;
            if (dep[v] != -1)
                continue;
            Q[rear++] = v;
            dep[v] = dep[u] + 1;
            gap[dep[v]]++;
        }
    }
}
int S[MAXN];
ll sap(int start, int end, int N)
{
    BFS(start, end);
    memcpy(cur, head, sizeof(head));
    int top = 0;
    int u = start;
    ll ans = 0;
    while (dep[start] < N)
    {
        if (u == end)
        {
            ll Min = INF;
            int inser;
            for (int i = 0; i < top; i++)
                if (Min > edge[S[i]].cap - edge[S[i]].flow)
                {
                    Min = edge[S[i]].cap - edge[S[i]].flow;
                    inser = i;
                }
            for (int i = 0; i < top; i++)
            {
                edge[S[i]].flow += Min;
                edge[S[i] ^ 1].flow -= Min;
            }
            ans += Min;
            top = inser;
            u = edge[S[top] ^ 1].to;
            continue;
        }
        bool flag = false;
        int v;
        for (int i = cur[u]; i != -1; i = edge[i].next)
        {
            v = edge[i].to;
            if (edge[i].cap - edge[i].flow && dep[v] + 1 == dep[u])
            {
                flag = true;
                cur[u] = i;
                break;
            }
        }
        if (flag)
        {
            S[top++] = cur[u];
            u = v;
            continue;
        }
        ll Min = N;
        for (int i = head[u]; i != -1; i = edge[i].next)
            if (edge[i].cap - edge[i].flow && dep[edge[i].to] < Min)
            {
                Min = dep[edge[i].to];
                cur[u] = i;
            }
        gap[dep[u]]--;
        if (!gap[dep[u]])
            return ans;
        dep[u] = Min + (ll)1;
        gap[dep[u]]++;
        if (u != start)
            u = edge[S[--top] ^ 1].to;
    }
    return ans;
}
bool spfa(int s, int t)
{
    queue<int> q;
    for (int i = 0; i < N; i++)
    {
        dis[i] = INF;
        vis[i] = false;
        pre[i] = -1;
    }
    dis[s] = 0;
    vis[s] = true;
    q.push(s);
    while (!q.empty())
    {
        int u = q.front();
        q.pop();
        vis[u] = false;
        for (int i = head[u]; i != -1; i = edge[i].next)
        {
            int v = edge[i].to;
            if (edge[i].cap > edge[i].flow && dis[v] > dis[u] + edge
                                                                    [i]
                                                                        .cost)
            {
                dis[v] = dis[u] + edge[i].cost;
                pre[v] = i;
                if (!vis[v])
                {
                    vis[v] = true;
                    q.push(v);
                }
            }
        }
    }
    if (pre[t] == -1)
        return false;
    else
        return true;
}
ll minCostMaxflow(int s, int t, ll &cost)
{
    ll flow = 0;
    cost = 0;
    while (spfa(s, t))
    {
        ll Min = INF;
        for (int i = pre[t]; i != -1; i = pre[edge[i ^ 1].to])
        {
            if (Min > edge[i].cap - edge[i].flow)
                Min = edge[i].cap - edge[i].flow;
        }
        for (int i = pre[t]; i != -1; i = pre[edge[i ^ 1].to])
        {
            edge[i].flow += Min;
            edge[i ^ 1].flow -= Min;
            cost += edge[i].cost * Min;
        }
        flow += Min;
    }
    return flow;
}
int n, m, k, s, t, u, v;
ll mn, mx, xx, yy, w, cc, cst;
int main()
{
    n = read();
    m = read();
    k = read();
    init(n);
    fo(i, 1, m)
    {
        u = read();
        v = read();
        u--;
        v--;
        w = read();
        cc = read();
        addedge(u, v, w, cc);
    }
    ll caosini = sap(0, n - 1, n);
    cout << caosini << " ";
    for (int i = 0; i < tol; i++)
        edge[i].flow = 0;
    int pp = tol;
    for (int i = 0; i < pp; i += 2)
    {
        addedge2(edge[i].u, edge[i].to, INF, edge[i].caonima);
    }
    addedge(n, 0, caosini + k, 0);
    N++;
    minCostMaxflow(n, n - 1, cst);
    cout << cst;

    return 0;
}
//10.图连通
//①floyd传递闭包
for (int k = 1; k <= n; k++)
{
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            if (k != i)
            {
                f(circle[i][k] && circle[k][j]) circle[i][j] = 1;
            }
        }
    }
}
//②判断正负环（还有一种办法是spfa，一个点入队次数>n则存在负权环）
//判定负环应用与很多二分答案后以是否存在负环来判定答案是否合法（题目多与环的权值或差分约束系统有关）
//首先一定要初始化为0；去当且仅当我们可以更新这个点的时候我们去遍历该点。若该点已被别遍历过（v==true)则找到负环。
//注意退出前一定要让当前节点的v变成false
//从每个点都要做一次dfs，但并不用每次都清空d数组，只在第一次遍历前清空d数组即可。
bool dfs(int now)
{
    int j;
    for (j = last[now]; j; j = next[j])
        if (d[now] + w[j] <= d[a[j]])
            if (v[a[j]])
            {
                v[now] = false; //退出前一定让v[now]=false，别忘了，也别写成v[a[j]]
                return true;
            }
            else
            {
                v[a[j]] = true;
                d[a[j]] = d[now] + w[j];
                if (dfs(a[j]))
                {
                    v[now] = false; //退出前一定让v[now]=false，别忘了，也别写成v[a[j]]
                    return true;
                }
            }
    v[now] = false; //退出前一定让v[now]=false，别忘了，也别写成v[a[j]]
    return false;
}
bool check()
{
    int i;
    memset(d, 0, sizeof(d)); //只需在第一次dfs前清空数组即可
    for (i = 1; i <= n; i++)
        if (dfs(i))
            return true;
    return false;
}
//③最小环
int floyd_circle()
{
    int ans = maxint;
    for (int k = 0; k < n; k++)
    {
        for (int i = 0; i <= k - 1; i++)
            for (int j = i + 1; j <= k - 1; j++)
                ans = min(ans, d[i][j] + block[i][k] + block[k][j]);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                d[i][j] = min(d[i][j], d[i][k] + d[k][j]);
    }
    return ans;
}
//④出边只有一条的最小环（信息传递）
void dfs(int x)
{
    s[top = 1] = x;
    while (top)
    {
        x = s[top];
        if (dep[x])
        {
            dep[x] = -1;
            --top;
            continue;
        }
        dep[x] = dep[s[top - 1]] + 1;
        if (!dep[to[x]])
        {
            s[top + 1] = to[x];
            ++top;
            continue;
        }
        else if (dep[to[x]] != -1 && ans > dep[x] - dep[to[x]] + 1)
            ans = dep[x] - dep[to[x]] + 1;
        dep[x] = -1;
        --top;
    }
}
for (i = 1; i <= n; ++i)
    if (!dep[i])
        dfs(i);
//④强连通分量
/*
[点连通度与边连通度] 在一个无向连通图中，如果有一个顶点集合，删除这个顶点集合，以
及这个集合中所有顶点相关联的边以后，原图变成多个连通块，就称这个点集为割点集合。
一个图的点连通度的定义为，最小割点集合中的顶点数。
类似的，如果有一个边集合，删除这个边集合以后，原图变成多个连通块，就称这个点集为
割边集合。一个图的边连通度的定义为，最小割边集合中的边数。
[双连通图、割点与桥]
如果一个无向连通图的点连通度大于 1，则称该图是点双连通的 (point biconnected)，简称
双连通或重连通。一个图有割点，当且仅当这个图的点连通度为 1，则割点集合的唯一元素
被称为割点 (cut point)，又叫关节点 (articulation point)。
如果一个无向连通图的边连通度大于 1，则称该图是边双连通的 (edge biconnected)，简称双
连通或重连通。一个图有桥，当且仅当这个图的边连通度为 1，则割边集合的唯一元素被称
为桥 (bridge)，又叫关节边 (articulation edge)。
可以看出，点双连通与边双连通都可以简称为双连通，它们之间是有着某种联系的，下文中
提到的双连通，均既可指点双连通，又可指边双连通。
[双连通分支]
在图 G 的所有子图 G’ 中，如果 G’ 是双连通的，则称 G’ 为双连通子图。如果一个双连
通子图 G’ 它不是任何一个双连通子图的真子集，则 G’ 为极大双连通子图。双连通分支
(biconnected component)，或重连通分支，就是图的极大双连通子图。特殊的，点双连通分
支又叫做块。 [求割点与桥]
该算法是 R.Tarjan 发明的。对图深度优先搜索，定义 DFS(u) 为 u 在搜索树（以下简称为
树）中被遍历到的次序号。定义 Low(u) 为 u 或 u 的子树中能通过非父子边追溯到的最早的
节点，即 DFS 序号最小的节点。根据定义，则有：
Low(u)=Min DFS(u) DFS(v) (u,v) 为后向边 (返祖边) 等价于 DFS(v)<DFS(u) 且 v 不为 u
的父亲节点 Low(v) (u,v) 为树枝边 (父子边) 一个顶点 u 是割点，当且仅当满足 (1) 或 (2)
(1) u 为树根，且 u 有多于一个子树。 (2) u 不为树根，且满足存在 (u,v) 为树枝边 (或称父子
边，即 u 为 v 在搜索树中的父亲)，使得 DFS(u)<=Low(v)。
一条无向边 (u,v) 是桥，当且仅当 (u,v) 为树枝边，且满足 DFS(u)<Low(v)。
[求双连通分支]
下面要分开讨论点双连通分支与边双连通分支的求法。
对于点双连通分支，实际上在求割点的过程中就能顺便把每个点双连通分支求出。建立一个
栈，存储当前双连通分支，在搜索图时，每找到一条树枝边或后向边 (非横叉边)，就把这条
边加入栈中。如果遇到某时满足 DFS(u)<=Low(v)，说明 u 是一个割点，同时把边从栈顶一
个个取出，直到遇到了边 (u,v)，取出的这些边与其关联的点，组成一个点双连通分支。割点
可以属于多个点双连通分支，其余点和每条边只属于且属于一个点双连通分支。
对于边双连通分支，求法更为简单。只需在求出所有的桥以后，把桥边删除，原图变成了多
个连通块，则每个连通块就是一个边双连通分支。桥不属于任何一个边双连通分支，其余的
边和每个顶点都属于且只属于一个边双连通分支。
[构造双连通图]
一个有桥的连通图，如何把它通过加边变成边双连通图？方法为首先求出所有的桥，然后删
除这些桥边，剩下的每个连通块都是一个双连通子图。把每个双连通子图收缩为一个顶点，
再把桥边加回来，最后的这个图一定是一棵树，边连通度为 1。
统计出树中度为 1 的节点的个数，即为叶节点的个数，记为 leaf。则至少在树上添加
(leaf+1)/2 条边，就能使树达到边二连通，所以至少添加的边数就是 (leaf+1)/2。具体方
法为，首先把两个最近公共祖先最远的两个叶节点之间连接一条边，这样可以把这两个点到
祖先的路径上所有点收缩到一起，因为一个形成的环一定是双连通的。然后再找两个最近公
共祖先最远的两个叶节点，这样一对一对找完，恰好是 (leaf+1)/2 次，把所有点收缩到了一
起。
*/
#include <iostream>
#include <cstdio>
#include <string>
#include <cstring>
#include <algorithm>
#include <stack>
#include <queue>
#include <vector>
using namespace std;
const int maxn = 500500;
struct edge
{
    int v;
    int nxt;
} e[maxn];
int n, m, s, p, mny[maxn], nmny[maxn];
int head[maxn], cnt;
int stop, sta[maxn], dfn[maxn], low[maxn], isin[maxn], indx;
int tot;
int d[maxn], vis[maxn];
bool inst[maxn];
stack<int> st;
vector<int> g[maxn];
int read()
{
    char ch = getchar();
    int x = 0, f = 1;
    while (!(ch >= '0' && ch <= '9'))
    {
        if (ch == '-')
            f = -1;
        ch = getchar();
    };
    while (ch >= '0' && ch <= '9')
    {
        x = x * 10 + (ch - '0');
        ch = getchar();
    };
    return x * f;
}
void ins(int u, int v)
{
    cnt++;
    e[cnt].v = v;
    e[cnt].nxt = head[u];
    head[u] = cnt;
}
void input()
{
    n = read();
    m = read();
    int u, v;
    for (int i = 1; i <= m; i++)
    {
        u = read();
        v = read();
        ins(u, v);
    }
    for (int i = 1; i <= n; i++)
        mny[i] = read();
    s = read();
    p = read();
}
void st_psh(int x)
{
    dfn[x] = low[x] = ++indx;
    inst[x] = true;
    sta[++stop] = x;
    st.push(x);
}
void tarjan(int x)
{
    int t;
    st_psh(x);
    while (!st.empty())
    {
        t = st.top();
        for (int i = head[t]; i; i = e[i].nxt)
        {
            if (dfn[e[i].v] == 0)
            {
                st_psh(e[i].v);
                break;
            }
        }
        if (t == st.top())
        {
            for (int i = head[t]; i; i = e[i].nxt)
            {
                if (dfn[e[i].v] > dfn[t])
                    low[t] = min(low[e[i].v], low[t]);
                else if (inst[e[i].v])
                {
                    low[t] = min(dfn[e[i].v], low[t]);
                }
            }
            if (dfn[t] == low[t])
            {
                ++tot;
                int j;
                do
                {
                    j = sta[stop--];
                    inst[j] = false;
                    isin[j] = tot;
                    nmny[tot] += mny[j];
                } while (j != t);
            }
            st.pop();
        }
    }
}
void spfa()
{
    int u, to;
    queue<int> q;
    u = isin[s];
    d[isin[s]] = nmny[isin[s]];
    vis[isin[s]] = true;
    q.push(isin[s]);
    while (!q.empty())
    {
        u = q.front();
        q.pop();
        for (int i = 0; i < g[u].size(); i++)
        {
            to = g[u][i];
            if (d[to] < d[u] + nmny[to])
            {
                d[to] = d[u] + nmny[to];
                if (!vis[to])
                {
                    vis[to] = true;
                    q.push(to);
                }
            }
        }
        vis[u] = false;
    }
}
void work()
{
    for (int i = 1; i <= n; i++)
    {
        if (!dfn[i])
            tarjan(i);
    }
    for (int i = 1; i <= n; i++)
    {
        for (int j = head[i]; j; j = e[j].nxt)
        {
            if (isin[i] != isin[e[j].v])
            {
                g[isin[i]].push_back(isin[e[j].v]);
                //cout<<isin[i]<<" "<<isin[e[j].v]<<endl;
            }
        }
    }
    spfa();
    int qs, ans = 0;
    while (p--)
    {
        qs = read();
        ans = max(d[isin[qs]], ans);
    }
    cout << ans;
}
int main()
{
    input();
    work();
    return 0;
}
//⑤割点与桥
/*
* 求无向图的割点和桥
* 可以找出割点和桥，求删掉每个点后增加的连通块。
* 需要注意重边的处理，可以先用矩阵存，再转邻接表，或者进行判重
*/
const int MAXN = 10010;
const int MAXM = 100010;
struct Edge
{
    int to, next;
    bool cut; //是否为桥的标记
} edge[MAXM];
int head[MAXN], tot;
int Low[MAXN], DFN[MAXN], Stack[MAXN];
int Index, top;
bool Instack[MAXN];
bool cut[MAXN];
int add_block[MAXN]; //删除一个点后增加的连通块
int bridge;
void addedge(int u, int v)
{
    edge[tot].to = v;
    edge[tot].next = head[u];
    edge[tot].cut = false;
    head[u] = tot++;
}
void Tarjan(int u, int pre)
{
    int v;
    Low[u] = DFN[u] = ++Index;
    Stack[top++] = u;
    Instack[u] = true;
    int son = 0;
    int pre_cnt = 0; //处理重边，如果不需要可以去掉
    for (int i = head[u]; i != -1; i = edge[i].next)
    {
        v = edge[i].to;
        if (v == pre && pre_cnt == 0)
        {
            pre_cnt++;
            continue;
        }
        if (!DFN[v])
        {
            son++;
            Tarjan(v, u);
            if (Low[u] > Low[v])
                Low[u] = Low[v];
            //桥
            //一条无向边 (u,v) 是桥，当且仅当 (u,v) 为树枝边，且满足
            DFS(u) < Low(v)。 if (Low[v] > DFN[u])
            {
                bridge++;
                edge[i].cut = true;
                edge[i ^ 1].cut = true;
            }
            //割点
            //一个顶点 u 是割点，当且仅当满足 (1) 或 (2) (1) u 为树根，且u 有多于一个子树。
            //(2) u 不为树根，且满足存在 (u,v) 为树枝边 (或称父子边，
            //即 u 为 v 在搜索树中的父亲)，使得 DFS(u)<=Low(v)
            if (u != pre && Low[v] >= DFN[u])
            { //不是树根
                cut[u] = true;
                add_block[u]++;
            }
        }
        else if (Low[u] > DFN[v])
            Low[u] = DFN[v];
    }
    //树根，分支数大于 1
    if (u == pre && son > 1)
        cut[u] = true;
    if (u == pre)
        add_block[u] = son - 1;
    Instack[u] = false;
    top--;
}
//1） UVA 796 Critical Links 给出一个无向图，按顺序输出桥
void solve(int N)
{
    memset(DFN, 0, sizeof(DFN));
    memset(Instack, false, sizeof(Instack));
    memset(add_block, 0, sizeof(add_block));
    memset(cut, false, sizeof(cut));
    Index = top = 0;
    bridge = 0;
    for (int i = 1; i <= N; i++)
        if (!DFN[i])
            Tarjan(i, i);
    printf("%d␣critical␣links\n", bridge);
    vector<pair<int, int>> ans;
    for (int u = 1; u <= N; u++)
        for (int i = head[u]; i != -1; i = edge[i].next)
            if (edge[i].cut && edge[i].to > u)
            {
                ans.push_back(make_pair(u, edge[i].to));
            }
    sort(ans.begin(), ans.end());
    //按顺序输出桥
    for (int i = 0; i < ans.size(); i++)
        printf("%d␣-␣%d\n", ans[i].first - 1, ans[i].second - 1);
    printf("\n");
}
void init()
{
    tot = 0;
    memset(head, -1, sizeof(head));
}
//处理重边
map<int, int> mapit;
inline bool isHash(int u, int v)
{
    if (mapit[u * MAXN + v])
        return true;
    if (mapit[v * MAXN + u])
        return true;
    mapit[u * MAXN + v] = mapit[v * MAXN + u] = 1;
    return false;
}
int main()
{
    int n;
    while (scanf("%d", &n) == 1)
    {
        init();
        int u;
        int k;
        int v;
        //mapit.clear();
        for (int i = 1; i <= n; i++)
        {
            scanf("%d (%d)", &u, &k);
            u++;
            //这样加边，要保证正边和反边是相邻的，建无向图
            while (k--)
            {
                scanf("%d", &v);
                v++;
                if (v <= u)
                    continue;
                //if(isHash(u,v))continue;
                addedge(u, v);
                addedge(v, u);
            }
        }
        solve(n);
    }
    return 0;
}
//2） POJ 2117 求删除一个点后，图中最多有多少个连通块
void solve(int N)
{
    memset(DFN, 0, sizeof(DFN));
    memset(Instack, 0, sizeof(Instack));
    memset(add_block, 0, sizeof(add_block));
    memset(cut, false, sizeof(cut));
    Index = top = 0;
    int cnt = 0; //原来的连通块数
    for (int i = 1; i <= N; i++)
        if (!DFN[i])
        {
            Tarjan(i, i); //找割点调用必须是 Tarjan(i,i)
            cnt++;
        }
    int ans = 0;
    for (int i = 1; i <= N; i++)
        ans = max(ans, cnt + add_block[i]);
    printf("%d\n", ans);
}
void init()
{
    tot = 0;
    memset(head, -1, sizeof(head));
}
int main()
{
    int n, m;
    int u, v;
    while (scanf("%d%d", &n, &m) == 2)
    {
        if (n == 0 && m == 0)
            break;
        init();
        while (m--)
        {
            scanf("%d%d", &u, &v);
            u++;
            v++;
            addedge(u, v);
            addedge(v, u);
        }
        solve(n);
    }
    return 0;
}
//⑥边双连通分量
//去掉桥，其余的连通分支就是边双连通分支了。一个有桥的连通图要变成边双连通图的话，
//把双连通子图收缩为一个点，形成一颗树。需要加的边为 (leaf+1)/2 (leaf 为叶子结点个数)
//POJ 3177 给定一个连通的无向图 G，至少要添加几条边，才能使其变为双连通图。
const int MAXN = 5010;  //点数
const int MAXM = 20010; //边数，因为是无向图，所以这个值要 *2
struct Edge
{
    int to, next;
    bool cut; //是否是桥标记
} edge[MAXM];
int head[MAXN], tot;
int Low[MAXN], DFN[MAXN], Stack[MAXN], Belong[MAXN]; //Belong 数组的值是
1 ∼ block int Index, top;
int block; //边双连通块数
bool Instack[MAXN];
int bridge; //桥的数目
void addedge(int u, int v)
{
    edge[tot].to = v;
    edge[tot].next = head[u];
    edge[tot].cut = false;
    head[u] = tot++;
}
void Tarjan(int u, int pre)
{
    int v;
    Low[u] = DFN[u] = ++Index;
    Stack[top++] = u;
    Instack[u] = true;
    int pre_cnt = 0;
    for (int i = head[u]; i != -1; i = edge[i].next)
    {
        v = edge[i].to;
        if (v == pre && pre_cnt == 0)
        {
            pre_cnt++;
            continue;
        }
        if (!DFN[v])
        {
            Tarjan(v, u);
            if (Low[u] > Low[v])
                Low[u] = Low[v];
            if (Low[v] > DFN[u])
            {
                bridge++;
                edge[i].cut = true;
                edge[i ^ 1].cut = true;
            }
        }
        else if (Instack[v] && Low[u] > DFN[v])
            Low[u] = DFN[v];
    }
    if (Low[u] == DFN[u])
    {
        block++;
        do
        {
            v = Stack[--top];
            Instack[v] = false;
            Belong[v] = block;
        } while (v != u);
    }
}
void init()
{
    tot = 0;
    memset(head, -1, sizeof(head));
}
int du[MAXN]; //缩点后形成树，每个点的度数
void solve(int n)
{
    memset(DFN, 0, sizeof(DFN));
    memset(Instack, false, sizeof(Instack));
    Index = top = block = 0;
    Tarjan(1, 0);
    int ans = 0;
    memset(du, 0, sizeof(du));
    for (int i = 1; i <= n; i++)
        for (int j = head[i]; j != -1; j = edge[j].next)
            if (edge[j].cut)
                du[Belong[i]]++;
    for (int i = 1; i <= block; i++)
        if (du[i] == 1)
            ans++;
    //找叶子结点的个数 ans, 构造边双连通图需要加边 (ans+1)/2
    printf("%d\n", (ans + 1) / 2);
}
int main()
{
    int n, m;
    int u, v;
    while (scanf("%d%d", &n, &m) == 2)
    {
        init();
        while (m--)
        {
            scanf("%d%d", &u, &v);
            addedge(u, v);
            addedge(v, u);
        }
        solve(n);
    }
    return 0;
}
//⑦点双联通分量
//对于点双连通分支，实际上在求割点的过程中就能顺便把每个点双连通分支求出。建立一个
//栈，存储当前双连通分支，在搜索图时，每找到一条树枝边或后向边 (非横叉边)，就把这条
//边加入栈中。如果遇到某时满足 DFS(u)<=Low(v)，说明 u 是一个割点，同时把边从栈顶一
//个个取出，直到遇到了边 (u,v)，取出的这些边与其关联的点，组成一个点双连通分支。割点
//可以属于多个点双连通分支，其余点和每条边只属于且属于一个点双连通分支。
//POJ 2942
//奇圈，二分图判断的染色法，求点双连通分支
/*
POJ 2942 Knights of the Round Table
亚瑟王要在圆桌上召开骑士会议，为了不引发骑士之间的冲突，
并且能够让会议的议题有令人满意的结果，每次开会前都必须对出席会议的骑士有如下要
求：
1、相互憎恨的两个骑士不能坐在直接相邻的 2 个位置；
2、出席会议的骑士数必须是奇数，这是为了让投票表决议题时都能有结果。
注意： 1、所给出的憎恨关系一定是双向的，不存在单向憎恨关系。
2、由于是圆桌会议，则每个出席的骑士身边必定刚好有 2 个骑士。
即每个骑士的座位两边都必定各有一个骑士。
3、一个骑士无法开会，就是说至少有 3 个骑士才可能开会。
首先根据给出的互相憎恨的图中得到补图。
然后就相当于找出不能形成奇圈的点。
利用下面两个定理：
（1）如果一个双连通分量内的某些顶点在一个奇圈中（即双连通分量含有奇圈），
那么这个双连通分量的其他顶点也在某个奇圈中；
（2）如果一个双连通分量含有奇圈，则他必定不是一个二分图。反过来也成立，这是一个
充要条件。
所以本题的做法，就是对补图求点双连通分量。
然后对于求得的点双连通分量，使用染色法判断是不是二分图，不是二分图，这个双连通分
量的点是可以存在的
*/
const int MAXN = 1010;
const int MAXM = 2000010;
struct Edge
{
    int to, next;
} edge[MAXM];
int head[MAXN], tot;
int Low[MAXN], DFN[MAXN], Stack[MAXN], Belong[MAXN];
int Index, top;
int block; //点双连通分量的个数
bool Instack[MAXN];
bool can[MAXN];
bool ok[MAXN];   //标记
int tmp[MAXN];   //暂时存储双连通分量中的点
int cc;          //tmp 的计数
int color[MAXN]; //染色
void addedge(int u, int v)
{
    edge[tot].to = v;
    edge[tot].next = head[u];
    head[u] = tot++;
}
//染色判断二分图
bool dfs(int u, int col)
{
    color[u] = col;
    for (int i = head[u]; i != -1; i = edge[i].next)
    {
        int v = edge[i].to;
        if (!ok[v])
            continue;
        if (color[v] != -1)
        {
            if (color[v] == col)
                return false;
            continue;
        }
        if (!dfs(v, !col))
            return false;
    }
    return true;
}
void Tarjan(int u, int pre)
{
    int v;
    Low[u] = DFN[u] = ++Index;
    Stack[top++] = u;
    Instack[u] = true;
    int pre_cnt = 0;
    for (int i = head[u]; i != -1; i = edge[i].next)
    {
        v = edge[i].to;
        if (v == pre && pre_cnt == 0)
        {
            pre_cnt++;
            continue;
        }
        if (!DFN[v])
        {
            Tarjan(v, u);
            if (Low[u] > Low[v])
                Low[u] = Low[v];
            if (Low[v] >= DFN[u])
            {
                block++;
                int vn;
                cc = 0;
                memset(ok, false, sizeof(ok));
                do
                {
                    vn = Stack[--top];
                    Belong[vn] = block;
                    Instack[vn] = false;
                    ok[vn] = true;
                    tmp[cc++] = vn;
                } while (vn != v);
                ok[u] = 1;
                memset(color, -1, sizeof(color));
                if (!dfs(u, 0))
                {
                    can[u] = true;
                    while (cc--)
                        can[tmp[cc]] = true;
                }
            }
        }
        else if (Instack[v] && Low[u] > DFN[v])
            Low[u] = DFN[v];
    }
}
void solve(int n)
{
    memset(DFN, 0, sizeof(DFN));
    memset(Instack, false, sizeof(Instack));
    Index = block = top = 0;
    memset(can, false, sizeof(can));
    for (int i = 1; i <= n; i++)
        if (!DFN[i])
            Tarjan(i, -1);
    int ans = n;
    for (int i = 1; i <= n; i++)
        if (can[i])
            ans--;
    printf("%d\n", ans);
}
void init()
{
    tot = 0;
    memset(head, -1, sizeof(head));
}
int g[MAXN][MAXN];
int main()
{
    int n, m;
    int u, v;
    while (scanf("%d%d", &n, &m) == 2)
    {
        if (n == 0 && m == 0)
            break;
        init();
        memset(g, 0, sizeof(g));
        while (m--)
        {
            scanf("%d%d", &u, &v);
            g[u][v] = g[v][u] = 1;
        }
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= n; j++)
                if (i != j && g[i][j] == 0)
                    addedge(i, j);
        solve(n);
    }
    return 0;
}
//11.2——SAT
//染色法
//HDU 1814
const int MAXN = 20020;
const int MAXM = 100010;
struct Edge
{
    int to, next;
} edge[MAXM];
int head[MAXN], tot;
void init()
{
    tot = 0;
    memset(head, -1, sizeof(head));
}
void addedge(int u, int v)
{
    edge[tot].to = v;
    edge[tot].next = head[u];
    head[u] = tot++;
}
bool vis[MAXN];   //染色标记，为 true 表示选择
int S[MAXN], top; //栈
bool dfs(int u)
{
    if (vis[u ^ 1])
        return false;
    if (vis[u])
        return true;
    vis[u] = true;
    S[top++] = u;
    for (int i = head[u]; i != -1; i = edge[i].next)
        if (!dfs(edge[i].to))
            return false;
    return true;
}
bool Twosat(int n)
{
    memset(vis, false, sizeof(vis));
    for (int i = 0; i < n; i += 2)
    {
        if (vis[i] || vis[i ^ 1])
            continue;
        top = 0;
        if (!dfs(i))
        {
            while (top)
                vis[S[--top]] = false;
            if (!dfs(i ^ 1))
                return false;
        }
    }
    return true;
}
int main()
{
    int n, m;
    int u, v;
    while (scanf("%d%d", &n, &m) == 2)
    {
        init();
        while (m--)
        {
            scanf("%d%d", &u, &v);
            u--;
            v--;
            addedge(u, v ^ 1);
            addedge(v, u ^ 1);
        }
        if (Twosat(2 * n))
        {
            for (int i = 0; i < 2 * n; i++)
                if (vis[i])
                    printf("%d\n", i + 1);
        }
        else
            printf("NIE\n");
    }
    return 0;
}
//强连通缩点（拓扑排序只能得到任意解）
//POJ 3648 Wedding
//******************************************
//2-SAT 强连通缩点
const int MAXN = 1010;
const int MAXM = 100010;
struct Edge
{
    int to, next;
} edge[MAXM];
int head[MAXN], tot;
void init()
{
    tot = 0;
    memset(head, -1, sizeof(head));
}
void addedge(int u, int v)
{
    edge[tot].to = v;
    edge[tot].next = head[u];
    head[u] = tot++;
}
int Low[MAXN], DFN[MAXN], Stack[MAXN], Belong[MAXN]; //Belong 数组的值 1∼ scc
int Index, top;
int scc;
bool Instack[MAXN];
int num[MAXN];
void Tarjan(int u)
{
    int v;
    Low[u] = DFN[u] = ++Index;
    Stack[top++] = u;
    Instack[u] = true;
    for (int i = head[u]; i != -1; i = edge[i].next)
    {
        v = edge[i].to;
        if (!DFN[v])
        {
            Tarjan(v);
            if (Low[u] > Low[v])
                Low[u] = Low[v];
        }
        else if (Instack[v] && Low[u] > DFN[v])
            Low[u] = DFN[v];
    }
    if (Low[u] == DFN[u])
    {
        scc++;
        do
        {
            v = Stack[--top];
            Instack[v] = false;
            Belong[v] = scc;
            num[scc]++;
        } while (v != u);
    }
}
bool solvable(int n) //n 是总个数, 需要选择一半
{
    memset(DFN, 0, sizeof(DFN));
    memset(Instack, false, sizeof(Instack));
    memset(num, 0, sizeof(num));
    Index = scc = top = 0;
    for (int i = 0; i < n; i++)
        if (!DFN[i])
            Tarjan(i);
    for (int i = 0; i < n; i += 2)
    {
        if (Belong[i] == Belong[i ^ 1])
            return false;
    }
    return true;
}
//*************************************************
//拓扑排序求任意一组解部分
queue<int> q1, q2;
vector<vector<int>> dag; //缩点后的逆向 DAG 图
char color[MAXN];        //染色，为'R' 是选择的
int indeg[MAXN];         //入度
int cf[MAXN];
void solve(int n)
{
    dag.assign(scc + 1, vector<int>());
    memset(indeg, 0, sizeof(indeg));
    memset(color, 0, sizeof(color));
    for (int u = 0; u < n; u++)
        for (int i = head[u]; i != -1; i = edge[i].next)
        {
            int v = edge[i].to;
            if (Belong[u] != Belong[v])
            {
                dag[Belong[v]].push_back(Belong[u]);
                indeg[Belong[u]]++;
            }
        }
    for (int i = 0; i < n; i += 2)
    {
        cf[Belong[i]] = Belong[i ^ 1];
        cf[Belong[i ^ 1]] = Belong[i];
    }
    while (!q1.empty())
        q1.pop();
    while (!q2.empty())
        q2.pop();
    for (int i = 1; i <= scc; i++)
        if (indeg[i] == 0)
            q1.push(i);
    while (!q1.empty())
    {
        int u = q1.front();
        q1.pop();
        if (color[u] == 0)
        {
            color[u] = 'R';
            color[cf[u]] = 'B';
        }
        int sz = dag[u].size();
        for (int i = 0; i < sz; i++)
        {
            indeg[dag[u][i]]--;
            if (indeg[dag[u][i]] == 0)
                q1.push(dag[u][i]);
        }
    }
}
int change(char s[])
{
    int ret = 0;
    int i = 0;
    while (s[i] >= '0' && s[i] <= '9')
    {
        ret *= 10;
        ret += s[i] - '0';
        i++;
    }
    if (s[i] == 'w')
        return 2 * ret;
    else
        return 2 * ret + 1;
}
int main()
{
    int n, m;
    char s1[10], s2[10];
    while (scanf("%d%d", &n, &m) == 2)
    {
        if (n == 0 && m == 0)
            break;
        init();
        while (m--)
        {
            scanf("%s%s", s1, s2);
            int u = change(s1);
            int v = change(s2);
            addedge(u ^ 1, v);
            addedge(v ^ 1, u);
        }
        addedge(1, 0);
        if (solvable(2 * n))
        {
            solve(2 * n);
            for (int i = 1; i < n; i++)
            {
                //注意这一定是判断 color[Belong]
                if (color[Belong[2 * i]] == 'R')
                    printf("%dw", i);
                else
                    printf("%dh", i);
                if (i < n - 1)
                    printf(" ");
                else
                    printf("\n");
            }
        }
        else
            printf("bad luck\n");
    }
    return 0;
}
//12.lca
//①倍增
//Gym - 101810M Greedy Pirate
#include <iostream>
#include <cstdio>
#include <string>
#include <cstring>
#include <algorithm>
#include <vector>
#define ll long long
#define fo(i, l, r) for (int i = l; i <= r; i++)
#define fd(i, l, r) for (int i = r; i >= l; i--)
#define mem(x) memset(x, 0, sizeof(x))
using namespace std;
const int maxn = 100050;
ll read()
{
    ll x = 0, f = 1;
    char ch = getchar();
    while (!(ch >= '0' && ch <= '9'))
    {
        if (ch == '-')
            f = -1;
        ch = getchar();
    };
    while (ch >= '0' && ch <= '9')
    {
        x = x * 10 + (ch - '0');
        ch = getchar();
    };
    return x * f;
}
struct edge
{
    int v;
    ll w;
    ll w2;
    int id;
    int next;
    friend bool operator<(edge a, edge b)
    {
        return a.w < b.w;
    }
} e[maxn * 3];
int cnt, head[maxn];
int n, q;
int uu, vv;
ll ww;
void ins(int u, int v, ll w, ll w2, int id)
{
    cnt++;
    e[cnt].v = v;
    e[cnt].w = w;
    e[cnt].w2 = w2;
    e[cnt].id = id;
    e[cnt].next = head[u];
    head[u] = cnt;
}
int deep[maxn], f[maxn][30];
ll dr[maxn], dr2[maxn], dsum[maxn];
void dfs(int x)
{
    int y, j, k;
    y = f[x][0];
    deep[x] = deep[y] + 1;
    for (k = 0; f[y][k] != 0; k++)
    {
        f[x][k + 1] = f[y][k];
        y = f[y][k];
    }
    for (int i = head[x]; i; i = e[i].next)
    {
        if (e[i].v == f[x][0])
            continue;
        dr[e[i].v] = dr[x] + e[i].w;
        dr2[e[i].v] = dr2[x] + e[i].w2;
        f[e[i].v][0] = x;
        dfs(e[i].v);
        dsum[x] += dsum[e[i].v] + e[i].w + e[i].w2;
    }
}
int findlca(int x, int y)
{
    int z, k, dd;
    if (deep[x] < deep[y])
    {
        z = x;
        x = y;
        y = z;
    }
    k = 0;
    for (dd = deep[x] - deep[y]; dd != 0; dd = dd >> 1)
    {
        if (dd & 1)
            x = f[x][k];
        k++;
    }
    if (x == y)
        return x;
    k = 0;
    while (k >= 0)
        if (f[x][k] != f[y][k])
        {
            x = f[x][k];
            y = f[y][k];
            k++;
        }
        else
            k--;
    return f[x][0];
}
void init()
{
    n = q = uu = vv = ww = cnt = 0;
    mem(deep);
    mem(dr);
    mem(dr2);
    mem(dsum);
    mem(f);
    mem(head);
}
ll get_dis(int u, int v)
{
    int l = findlca(u, v);
    ll ans = 0;
    ans = dsum[1] - dr[u] - dr2[v] + dr[l] + dr2[l];
    return ans;
}
int main()
{
    int T;
    cin >> T;
    int u, v;
    ll w, w2, ans;
    while (T--)
    {
        init();
        n = read();

        fo(i, 1, n - 1)
        {
            u = read();
            v = read();
            w = read();
            w2 = read();
            ins(u, v, w, w2, i);
            ins(v, u, w2, w, i);
        }
        q = read();
        dfs(1);
        fo(i, 1, q)
        {
            u = read();
            v = read();
            ans = get_dis(u, v);
            printf("%I64d\n", ans);
        }
    }
    return 0;
}
//②tarjan
/*
* POJ 1470
* 给出一颗有向树， Q 个查询
* 输出查询结果中每个点出现次数
*/
/*
* 离线算法， LCATarjan
* 复杂度O(n+Q);
*/
const int MAXN = 1010;
const int MAXQ = 500010; //查询数的最大值
//并查集部分
int F[MAXN]; //需要初始化为 -1
int find(int x)
{
    if (F[x] == -1)
        return x;
    return F[x] = find(F[x]);
}
void bing(int u, int v)
{
    int t1 = find(u);
    int t2 = find(v);
    if (t1 != t2)
        F[t1] = t2;
}
//************************
bool vis[MAXN];     //访问标记
int ancestor[MAXN]; //祖先
struct Edge
{
    int to, next;
} edge[MAXN * 2];
int head[MAXN], tot;
void addedge(int u, int v)
{
    edge[tot].to = v;
    edge[tot].next = head[u];
    head[u] = tot++;
}
struct Query
{
    int q, next;
    int index; //查询编号
} query[MAXQ * 2];
int answer[MAXQ]; //存储最后的查询结果，下标 0 Q-1
int h[MAXQ];
int tt;
int Q;
void add_query(int u, int v, int index)
{
    query[tt].q = v;
    query[tt].next = h[u];
    query[tt].index = index;
    h[u] = tt++;
    query[tt].q = u;
    query[tt].next = h[v];
    query[tt].index = index;
    h[v] = tt++;
}
void init()
{
    tot = 0;
    memset(head,-1, sizeof(head));
    tt = 0;
    memset(h,-1, sizeof(h));
    memset(vis, false, sizeof(vis));
    memset(F,-1, sizeof(F));
    memset(ancestor, 0, sizeof(ancestor));
}
void LCA(int u)
{
    ancestor[u] = u;
    vis[u] = true;
    for (int i = head[u]; i != -1; i = edge[i].next)
    {
        int v = edge[i].to;
        if (vis[v])
            continue;
        LCA(v);
        bing(u, v);
        ancestor[find(u)] = u;
    }
    for (int i = h[u]; i != -1; i = query[i].next)
    {
        int v = query[i].q;
        if (vis[v])
        {
            answer[query[i].index] = ancestor[find(v)];
        }
    }
}
bool flag[MAXN];
int Count_num[MAXN];
int main()
{
    int n;
    int u, v, k;
    while (scanf("%d", &n) == 1)
    {
        init();
        memset(flag, false, sizeof(flag));
        for (int i = 1; i <= n; i++)
        {
            scanf("%d:(%d)", &u, &k);
            while (k--)
            {
                scanf("%d", &v);
                flag[v] = true;
                addedge(u, v);
                addedge(v, u);
            }
        }
        scanf("%d", &Q);
        for (int i = 0; i < Q; i++)
        {
            char ch;
            cin >> ch;
            scanf("%d␣%d)", &u, &v);
            add_query(u, v, i);
        }
        int root;
        for (int i = 1; i <= n; i++)
            if (!flag[i])
            {
                root = i;
                break;
            }
        LCA(root);
        memset(Count_num, 0, sizeof(Count_num));
        for (int i = 0; i < Q; i++)
            Count_num[answer[i]]++;
        for (int i = 1; i <= n; i++)
            if (Count_num[i] > 0)
                printf("%d:%d\n", i, Count_num[i]);
    }
    return 0;
}
