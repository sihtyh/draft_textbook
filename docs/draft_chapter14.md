# Python 金融建模：基础与应用

MIT Licensed | Copyright © 2024-present by [Yun Liao ](mailto:james@x.cool)

## Python 应用篇（第13-16章）

第13章：金融风险建模和分析

第14章：效率分析模型

第15章：信用评级和信用评分模型

第16章：量化投资策略算法和应用

### 第14章：效率分析模型

#### 14.1 效率分析

虽然效率分析在产业，投资机会，生产力分析中都有很多应用，本书以金融机构为例，我们进行效率分析方面的一些评估模型的学习和如何使用python语言实现相关模型的计算和验证。

**金融机构效率**

在金融机构的背景下，效率是指最大化利润同时最小化成本、风险和监管负担。高效的金融机构能够优化其运营、简化流程并充分利用技术来提高表现。

**金融机构效率的类型**

1. **操作效率** ：指减少成本、简化流程和提高生产力的能力。
2. **财务效率** ：关系到金融资源的优化，如资本配置、流动性管理和风险管理。
3. **技术效率** ：涉及到使用技术来改善运营、减少成本并提高客户体验。

**评估金融机构效率的指标**

1. **股东权益回报率（ROE）** ：衡量利润相对于股东权益的百分比。
2. **费用收入比** ：计算运营费用相对于收入的百分比。
3. **资产利用率** ：评估资产，如贷款、存款和投资的有效使用情况。
4. **操作杠杆** ：评估增加收入同时控制成本的能力。
5. **数字经济的影响** ：衡量数字渠道在客户交互和交易中的应用程度。

**金融机构效率面临的挑战**

1. **监管合规** ：遵守复杂的监管要求可能是耗时和昂贵的。
2. **旧系统** ：过时的技术基础设施可能会阻碍效率和创新。
3. **客户行为变化** ：适应不断变化的客户偏好和期望可能是挑战性的。
4. **金融科技公司竞争** ：金融机构面临来自金融科技公司的竞争，这些公司通常更具灵活性和高效。

效率分析是在金融、经济和管理科学中评估组织、公司或决策单位（DMU）的表现的关键工具。两种主要的效率分析技术是数据Envelope 分析（DEA）和随机 Frontier 分析（SFA）。

##### 1. 数据Envelope 分析（DEA）

DEA 是一種非参数方法，使用线性规划来评估 DMU 的效率。它最先由 Charnes、Cooper 和 Rhodes 在 1978 年引入。

**主要特征：**

* DEA 计算每个 DMU 相对于理想或高效前沿的相对效率。
* 它可以同时处理多个输入和输出。
* 方法对输入和输出的选择非常敏感，同时也对数据质量非常敏感。
* DEA 可以识别出低效的 DMU，并提供改进的见解。

**DEA 模型类型：**

* 输入导向 DEA：在保持输出不变的情况下最小化输入。
* 输出导向 DEA：在保持输入不变的情况下最大化输出。
* 非导向 DEA：同时最小化输入和最大化输出。

##### 2. 随机 Frontier 分析（SFA）

SFA 是一種参数方法，使用经济计量技术来估计最佳实践的 Frontier。它最先由 Aigner、Lovell 和 Schmidt 在 1977 年引入。

**主要特征：**

* SFA 假设生产函数或成本函数具有特定的函数形式。
* 方法使用最大似然估计来估计低效项。
* SFA 可以处理多个输入和输出，同时也可以处理随机误差和离群值。
* 它提供技术效率、配置效率和规模效率的估计。

**SFA 模型类型：**

* 生产 Frontier 模型：根据一组输入来估计可能的最大输出。
* 成本 Frontier 模型：根据一组输出来估计可能的最小成本。
* 距离函数模型：根据观察到的生产或成本水平来估计与 Frontier 之间的距离。

**DEA 和 SFA 的比较：**

* DEA 是非参数的，而 SFA 是参数的。
* DEA 在处理多个输入和输出时更加灵活，而 SFA 假设特定的函数形式。
* DEA 提供相对效率分数，而 SFA 估计绝对效率水平。
* 两种方法都可以识别出低效的 DMU，但 SFA 提供了更多关于低效来源的见解。

总之，DEA 和 SFA 都是效率分析的强大工具，每一种都有其优缺点。选择这些技术取决于特定的研究问题、数据特征和结果的细节要求。

#### 14.2 DEA模型

DEA 是一种非参数方法，用于评估决策单元（DMU）的相对效率。

最早期的DEA 模型（Charnes、Cooper 和 Rhodes 1978）基于以下假设：

**凸性假设** ：生产可能性集（PPS）被假设为凸性的，这意味着如果两个DMU是高效的，则它们之间的任何线性组合也将是高效的。

**单调性假设** ：输入和输出之间被假设为单调关系，这意味着输入的增加将导致相应输出的增加。

**强可抛弃假设** ：PPS 被假设为强可抛弃的，这意味着如果一个DMU是高效的，那么它可以被缩放或扩展，而不会影响其高效性。

**自由可抛弃（FDH-Free disposable Hypothesis）假设** ：FDH假设是强可抛弃假设的一个弱版本，它允许输入和输出在某些情况下进行缩放。

**常数回报尺度（CRS-Constant Return to Scale）假设** ：CRS假设意味着生产过程具有常数回报尺度，这意味着所有输入的增加将导致输出的相应增加。

DEA 模型的基本结构包括：

* 输入（x）：DMU 使用的资源或因素，以生产输出。
* 输出（y）：DMU 使用输入产生的结果或产出。
* 效率：衡量每个 DMU 如何高效地使用其输入来生产输出。

DEA 模型可以根据不同的目的和假设选择不同的形式，例如：

* 输入导向模型：在保持输出不变的情况下最小化输入。
* 输出导向模型：在保持输入不变的情况下最大化输出。
* 非导向模型：同时最小化输入和最大化输出。

##### 14.2.1CCR模型:输入导向 DEA 模型

输入导向 DEA 模型，也称为输入最小化 DEA 模型，其目标是保持当前的输出水平的情况下最小化输入使用量。该模型在减少成本或节约资源而不牺牲性能的情况下非常有用。

**数学公式**
假设我们有一组 `n` 个决策单位（DMU），每个 DMU 使用 `m` 个输入特征，生产 `s` 个输出特征，X_i(1-m)代表各输入特征的单位，Y_i(1-s)代表各输出特征的单位  。输入导向 DEA 模型可以被公式化为：

 **最小化** : $θ = (∑xi / yi)$
约束条件：

* $∑λj xij ≤ θ xi, i = 1, ..., m$
* $∑λj yij ≥ yi, j = 1, ..., s$
* $λj ≥ 0, j = 1, ..., n$
* $∑λj = 1$

其中：

* `xi` 是 DMU `i` 的输入使用量
* `yi` 是 DMU `i` 的输出生产量

下面是一个使用 Python 实现输入导向 DEA 模型的示例：

这个代码定义了一个简单的输入导向 DEA 模型，具有 5 个 DMU、2 个输入和 1 个输出。输入数据被定义为一个 2D 数组，其中每行代表一个 DMU，每列代表一个输入。输出数据被定义为一个 2D 数组，其中每行代表一个 DMU，每列代表一个输出。DEA 模型参数被定义，包括效率目标$\theta$。然后，线性规划问题被定义使用 SciPy 的 `linprog` 函数，这将求解最优 lambda 值，以便在保持当前输出水平的情况下最小化输入使用量。最后，该代码提取了最优 lambda 值，并计算每个 DMU 的效率分数，通过 lambda 值和输出数据的点积除以输入数据的总和。

```python
import numpy as np
from scipy.optimize import linprog
# 定义 DMU 的数量、输入和输出
n_dmus = 5
n_inputs = 2
n_outputs = 1
# 定义输入和输出数据
inputs = np.array([[10, 20], [15, 30], [12, 25], [18, 35], [11, 22]])
outputs = np.array([[100], [120], [110], [130], [105]])
# 定义 DEA 模型参数
theta = 1  # 效率目标
# 定义线性规划问题
A_ub = np.vstack((np.eye(n_dmus), -np.eye(n_dmus)))
b_ub = np.concatenate((inputs, inputs))
A_eq = np.array([[1]*n_dmus])
b_eq = [1]
c = -outputs.flatten()
res = linprog(c, A_ub, b_ub, A_eq, b_eq, method="highs")
# 提取最优 lambda 值
lambda_opt = res.x
# 计算效率分数
efficiency_scores = np.dot(lambda_opt, outputs) / inputs.sum(axis=0)
print("Efficiency Scores:", efficiency_scores)
```

##### 14.2.2 BCC模型

Banker、Charnes 和 Cooper (BCC) 模型是一种数据包络分析（DEA）模型，它使用线性规划方法来评估决策单元（DMU）的效率。它是 DEA 模型中最广泛使用的模型之一，特别适用于评价具有多个输入和输出的组织机构的性能。BCC 模型假设生产过程可以用观测数据点的凸包表示，每个 DMU 由其输入和输出特征。模型通过比较每个 DMU 与最佳实践前沿来评估效率，最佳实践前沿代表了在给定输入下可达到的最大输出。

**BCC 模型的具体数学构型**

输入方向 BCC 模型：

$\max \theta$
$s. t.\quad \sum_{j=1}^n \lambda_j x_ij = \theta x_i0 \quad i=1,...,m$
$\sum_{j=1}^n \lambda_j y_rj \geq y_r0 \quad r=1,...,s$
$\lambda_j \geq 0 \quad j=1,...,n$

输出方向 BCC 模型：

$\min \phi$
$s. t.\quad \sum_{j=1}^n \lambda_j x_ij \leq x_i0 \quad i=1,...,m$
$\sum_{j=1}^n \lambda_j y_rj = \phi y_r0 \quad r=1,...,s$
$\lambda_j \geq 0 \quad j=1,...,n$

其中：

* $x_ij$ 是 DMU j 使用的输入 i 的数量
* $y_rj$是 DMU j 生产的输出 r的数量
* $x_i0$和 $y_r0$ 是被评估 DMU（DMU 0）的输入和输出
* $\theta$和 $\phi$ 是效率分数
* $\lambda_j$ 是参考集中的每个 DMU 的权重

BCC 模型通过构建一个线性规划问题来最大化（或最小化）被评估 DMU 的效率分数， 以确保效率分数是相对于最佳实践前沿计算的。在输入方向 BCC 模型中，目标是最大化效率分数 $\theta$，它代表了可以减少的输入比例，同时保持相同的输出水平。约束确保被评估 DMU 的输入减少了一个因子 $\theta$，同时保持与最佳实践前沿相同的输出水平。在输出方向 BCC 模型中，目标是最小化效率分数 $\phi$，它代表了可以增加的输出比例，同时保持相同的输入水平。约束确保被评估 DMU 的输出增加了一个因子 $\phi$，同时保持与最佳实践前沿相同的输入水平。

CCR模型和BCC模型的关键区别在于：

1. CCR 模型是输入导向的，意思是它集中于在保持输出不变的情况下最小化输入。而BCC模型既有输入导向也有输出导向
2. CCR 假设恒定规模回报CRS，而 BCC 允许可变规模回报VRS
3. CCR 不允许闲置变量，而 BCC 允许闲置变量。
4. CCR 衡量技术效率和配置效率即TE 和 AE，而 BCC模型除了衡量TE和AE之外还可以度量规模效率即SE。

##### 14.2.3 基于松弛条件的DEA模型：Slacks-Based Measure of Efficiency (SBM)

基于松弛条件的DEA模型（Slacks-Based Measure，简称SBM），是一种数据包络分析（DEA）模型，用来评估决策单元（DMU）的效率。该模型由Tone在2001年首次提出。

**关键特征：**

* **松弛变量** ：SBM模型引入松弛变量来捕捉DMU的输入和输出中的不效率。
* **方向距离函数** ：模型使用方向距离函数来衡量DMU的效率，这允许同时优化多个输入和输出。
* **非径向** ：与径向DEA模型不同，SBM模型是非径向的，即不需要所有输入的比例减少或所有输出的比例增加。

**数学公式：**
SBM模型可以公式化为：

$\max \frac{1 + \frac{1}{m} \sum_{i=1}^m \frac{s_i^-}{x_{io}}} {1 - \frac{1}{s} \sum_{r=1}^s \frac{s_r^+}{y_{ro}}}$

subject to:

* $\sum_{j=1}^n \lambda_j x_ij + s_i^- = x_io, i = 1, ..., m$
* $\sum_{j=1}^n \lambda_j y_rij - s_r^+ = y_ro, r = 1, ..., s$
* $s_i^- \geq 0, i = 1, ..., m$
* $s_r^+ \geq 0, r = 1, ..., s$
* $\lambda_j \geq 0, j = 1, ..., n$
  其中：
* $x_ij$和$y_rij$分别是DMUj**j**的输入和输出
* $x_io$和$y_ro$分别是目标值的输入和输出
* $s_i^-$和$s_r^+$分别是输入和输出的松弛变量
* $\lambda_j$是分配给DMUj的权重
* m和s分别是输入和输出的数量

**解释：**
SBM模型通过计算DMU的输入和输出中的最大可能改进来衡量其效率，同时保持相同的输出质量。基于松弛条件的方法可以识别出特定的输入或输出无效，为改进建议提供了有价值的信息。

**优点：**

* **更真实地表示** ：SBM模型可以捕捉输入和输出之间更加复杂的关系，因为它不需要所有输入的比例减少或所有输出的比例增加。
* **识别 inefficiencies** ：基于松弛条件的方法有助于识别特定的无效率，为改进建议提供了有价值的信息

##### 14.2.4 超效率DEA模型: Super-Efficiency DEA

超效率DEA模型，也被称为Andersson和Boone（2009）模型，是一种数据包络分析（DEA）模型，通过允许高于1的超效率分数来评估决策单元（DMU）的效率。

**关键特征：**

* **超效率** ：该模型允许DMU具有高于1的效率分数，表明它们比最佳实践前沿运营更为高效。
* **修改方向距离函数** ：该模型使用修改后的方向距离函数来衡量DMU的效率，这使得超效率分数的计算成为可能。

**数学公式：**
超效率DEA模型形式 为：

$\min \phi = \frac{1 + \frac{1}{m} \sum_{i=1}^m \frac{s_i^-}{x_{io}}} {1 - \frac{1}{s} \sum_{r=1}^s \frac{s_r^+}{y_{ro}}}$
Subject to:

* $\sum_{j=1}^n \lambda_j x_{ij} - x_{i0}\phi_i \leq x_io, i = 1, ..., m$
* $\sum_{j=1}^n \lambda_j y_rij + y_{r0}\phi_r \geq y_ro, r = 1, ..., s$
* $s_i^- \geq 0, i = 1, ..., m$
* $s_r^+ \geq 0, r = 1, ..., s$
* $\lambda_j \geq 0, j = 1, ..., n$
* 其中：
* $x_ij$和$y_rij$分别是DMUj**j**的输入和输出
* $x_io$和$y_ro$分别是目标值的输入和输出
* $s_i^-$和$s_r^+$分别是输入和输出的松弛变量
* $\lambda_j$是分配给DMUj的权重
* m和s分别是输入和输出的数量

**解释：**
超效率DEA模型允许DMU具有高于1的效率分数，表明它们比最佳实践前沿运营更为高效。这使得“超效率”DMU的识别成为可能，这些DMU在其同行中表现出色。

**优点：**

* **识别超效率DMU** ：该模型使得超效率DMU的识别成为可能。
* **提供更为细腻的效率视图** ：通过允许超效率分数，该模型提供了一个更为细腻和详细的效率性能视图。

#### 14.3 DEA模型Python实现

##### 示例1：CCR/BCC/SBM DEA实现

以下是CCR/BCC/SBM的DEA模型实现代码, 对多个DMU某年的数据进行求解，输入变量为D,N，输出变量为O,I

```python
import numpy as np
import pandas as pd
import scipy.optimize as op
import os

def baseff(input_variable,desirable_output,dmu,data, method='highs'):
    """用于求解ccr/bcc/sbm模型  
    Parameters:
    -----------
    input_variable:
        投入[v1,v2,v3,...] 
    desirable_output:
        期望产出[v1,v2,v3,...]
    dmu:
        决策单元
    data:
        主数据
    method:
        求解方法.默认'highs',可选'highs-ds','highs-ipm'
	Return:
        ---
	res : DataFrame
		结果数据框[dmu	TE]
    """ 
    res = pd.DataFrame(columns = ['dmu','TE_CCR','TE_BCC','TE_SBM'], index = data.index)
    res['dmu'] = data[dmu]   
    res['TE_CCR']=np.nan
    res['TE_BCC']=np.nan
    res['TE_SBM']=np.nan
  
    d1=data[input_variable].T
    d2=-data[desirable_output].T 
    ## lambda有dmu个数个，S有变量个数个
    dmu_counts = data.shape[0]
     ## 投入个数
    m = len(input_variable)
    ## 期望产出个数
    s = len(desirable_output)
    n = dmu_counts  
    cols = input_variable+desirable_output
    newcolsccr = []
    newcolssbm = []
  
    for j in cols:
        newcolsccr.append(j+'_coef')
        res[j+'_coef'] = np.nan  
    for j in cols:
        newcolssbm.append(j+'_slack')
        res[j+'_slack'] = np.nan
 
    for i in range(dmu_counts):  
        ## 优化目标函数的系数矩阵
        c = [0]*m + list(-data.loc[i, desirable_output])  
        A_eq = [list(data.loc[i,input_variable]) + [0]*s]
        b_eq = 1  
        A_ub = []
        for j1 in range(dmu_counts):
            ub1 = list(-data.loc[j1,input_variable]) + list(data.loc[j1,desirable_output])  
            A_ub.append(ub1)   
        b_ub = [0]* dmu_counts  
        bounds = [(0, None)]* (m+s) 
        ## 求解
        op1 = op.linprog(c = c,A_eq=A_eq,b_eq=b_eq,A_ub=A_ub,b_ub=b_ub,bounds=bounds, method = method)
        res.loc[i, 'ops'] = op1.fun
        res.loc[i, newcolsccr] =op1.x
        w =   op1.x[m:m+s]
        theta =np.dot(data.loc[i,desirable_output],w)
        res.loc[i,'TE_CCR'] =theta
  
    for i in range(dmu_counts):   
        A_ub = []
        c = [0] * n + [1]  
        A_eq = [[1] * n + [0]]  
        b_eq = 1   
        d1['add']= -d1.loc[:,i] 
        d2['add']= 0
        ub=pd.concat([d1,d2])
        ub1= ub.values.tolist()
        A_ub = ub1  
        b_ub = [0]* m + list(-data.loc[i,desirable_output])  
        bounds = [(0,None)]*(n+1)
        ## 求解
        op2 = op.linprog(c = c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,
                         bounds=bounds, method = method)
        res.loc[i, 'TE_BCC'] = op2.fun  

    for i in range(dmu_counts):  
        ## 优化目标
        c = [0] * dmu_counts + [1] +  list(-1 / (m * data.loc[i, input_variable])) + [0] * s  
        ## 约束条件
        A_eq = [[0] * dmu_counts + [1] + [0] * m  + list(1/(s * data.loc[i, desirable_output]))]  
        ## 约束条件（1）
        for j1 in range(m):
            list1 = [0] * m
            list1[j1] = 1
            eq1 = list(data[input_variable[j1]]) + [-data.loc[i ,input_variable[j1]]] + list1 + [0] * s
            A_eq.append(eq1)
        ## 约束条件（2）
        for j2 in range(s):
            list2 = [0] * s
            list2[j2] = -1
            eq2 = list(data[desirable_output[j2]]) + [-data.loc[i, desirable_output[j2]]] + [0] * m + list2 
            A_eq.append(eq2)

        b_eq = [1] + [0] * (m + s)   
        bounds = [(0, None)]*(n + m + s + 1)
        ## 求解
        op3 = op.linprog(c = c,A_eq=A_eq,b_eq=b_eq,bounds=bounds,method = method)
        res.loc[i, 'TE_SBM'] = op3.fun
        res.loc[i, newcolssbm] = op3.x[dmu_counts+1 :]   
  
    return res
os.chdir('E:\\valdata\Data_input') 
data = pd.read_excel('test.xlsx')
basic_eff = baseff(input_variable=['D','N'], 
               desirable_output=['O','I'], 
                dmu = ['dmu'], data = data)

### 面板数据
### 全局 : data = 所有的样本
### 每年 : data = data[data["year"] == 某一年,""]，对每年分别求TE

```

##### 示例2：超效率DEA

以下是超效率DEA模型实现代码, 对多个DMU某年的数据进行求解，输入变量为D,N，输出变量为O,I

```python
import numpy as np
import pandas as pd
import scipy.optimize as op
import os
def super_sbm_nobad_gen(data,in_var,out_var,dmu,method):
    res = pd.DataFrame(columns = ['dmu','eff'], index = data.index)
    res['dmu'] = data['dmu']
     ## lambda有dmu个数个，S有变量个数个
    dmu_counts = data.shape[0]
         ## 投入个数
    m = len(in_var)
    ## 期望产出个数
    s = len(out_var)  
    n = dmu_counts  
    total = dmu_counts + m + s + 1
    cols = in_var + out_var
    coefcols= []
    slackcols= []
    newcols= []
    for j in cols:
        newcols.append(j+'_slack')
        res[j+'_slack'] = np.nan
    for i in range(dmu_counts):
        datai = data.drop([i])
        ## 优化目标
        c = [0] * (n-1) + [1] +  list( 1 / (m * data.loc[i, in_var])) + [0] * s 
        ## 约束条件
        A_eq = [[0] *(n-1) + [1] + [0] * m  + [-1/s]*s ]
        eq1 = [1] *(n-1) + [-1] + [0] * (m+s) 
        A_eq.append(eq1)
        ## 约束条件（1）
        A_ub = []
        for j1 in range(m):
            list1 = [0] * m
            list1[j1] = -1
            ub1 = list(datai[in_var[j1]]) + [-data.loc[i,in_var[j1]]] + list1 + [0] * s
            A_ub.append(ub1)
        ## 约束条件（2）
        for j2 in range(s):
            list2 = [0] * s
            list2[j2] = -data.loc[i, out_var[j2]]
            ub2 = list(-datai[out_var[j2]]) +[data.loc[i,out_var[j2]]] + [0] * m + list2 
            A_ub.append(ub2)
        ## 约束条件（3）   
        b_eq = [1]+[0]
        b_ub = [0]* (m+s)   
        bounds = [(0, None)]*(n+m+s)
        ## 求解
        op1 = op.linprog(c = c,A_eq=A_eq,b_eq=b_eq, A_ub=A_ub,b_ub=b_ub,bounds=bounds,method = method)
        res.loc[i, 'eff'] = op1.fun  
        res.loc[i, newcols] = op1.x[n:]  
        coefcols.append(op1.x)


    for i in range(dmu_counts):
        datai = data.drop([i])
        ## 优化目标
        c = [0] * (n-1) + [1] +  list( -1 / (m * data.loc[i, in_var])) + [0] * s   
        ## 约束条件
        A_eq = [[0] *(n-1) + [1] + [0] * m  + [1/s]*s]
        eq1 = [1] *(n-1) + [-1] + [0] * (m+s) 
        A_eq.append(eq1)
        ## 约束条件（1）
        for j1 in range(m):
            list1 = [0] * m
            list1[j1] = 1
            ub1 = list(datai[in_var[j1]]) + [-data.loc[i,in_var[j1]]] + list1 + [0] * s
            A_eq.append(ub1)

        ## 约束条件（2）
        for j2 in range(s):
            list2 = [0] * s
            list2[j2] = -data.loc[i, out_var[j2]]
            ub2 = list(datai[out_var[j2]]) + [-data.loc[i,out_var[j2]]] + [0] * m + list2 
            A_eq.append(ub2)

        ## 约束条件（3）   
        b_eq = [1] +[0]* (m+s+1)  
        bounds = [(0, None)]*(n+m+s)
        ## 求解
        op2 = op.linprog(c = c,A_eq=A_eq,b_eq=b_eq,bounds=bounds,method = method)
        if op2.fun!=None:
            res.loc[i, 'eff'] = op2.fun
            res.loc[i, newcols] = op2.x[n:] 
            coefcols.append(op2.x)
    return res


os.chdir('E:\\valdata\Data_input') 
data = pd.read_excel('test.xlsx')
in_var = ['D','N']
out_var = ['O','I']
dmu = 'dmu'
res1 = super_sbm_nobad_gen(data,in_var,out_var,dmu,method='highs')
```

#### 14.4 随机前沿分析：Stochastic Frontier Analysis模型

随机前沿分析是一种统计方法，用于在存在随机噪声和不可观察因素的情况下分析决策单元（DMU）的效率。它是传统确定性前沿分析模型的扩展，这些模型假设所有偏离前沿的都是由于不效率引起的。

**SFA 模型**
SFA 模型假设生产过程受到两种类型的扰动：

1. **非效率** ：这是指 DMU 的输出从其潜在最大输出的偏离，可能是由于管理不当、资源不足或技术不佳等因素引起的。
2. **随机噪声** ：这是指来自不可预测和不可控的事件的输出变化，例如气候条件、设备故障或其他随机事件。
   SFA 模型可以公式化为：

$y_i = \alpha + \beta x_i + v_i - u_i$
其中：

* $y_i$ 是 DMU **i** 的观测输出
* $\alpha$ 是截距项
* $\beta$ 是输入变量 $x_i$ 的系数向量
* $v_i$ 是随机噪声项，假设为正态分布，均值为 0，方差为 $\sigma_v^2$
* $u_i$ 是不效率项，假设为非负指数分布，参数为 **λ**

**估计**
SFA 模型参数通常使用最大似然估计（MLE）或贝叶斯方法进行估计。

**优点**

1. **处理随机噪声** ：SFA 模型明确地考虑了生产过程中的随机噪声，这可以导致不效率估计的更高精度。
2. **允许异方差** ：SFA 模型允许不同 DMU 的输出存在不同的可变性水平。
3. **提供不效率度量** ：SFA 模型提供了不效率项 u_i**u**i 的估计，这可以用于排名 DMU 按照相对效率。

##### 示例3：在python中使用R 语言调用SFA模型进行分析

在本案例中，我们学习如何在python中调用其他语言，特别是一个重要的统计软件R的应用全过程方法。

下载安装rpy2

![1720875506415](https://file+.vscode-resource.vscode-cdn.net/d%3A/yunpan/%E5%B7%A5%E4%BD%9C-%E5%88%9B%E6%96%B0%E7%A7%91%E7%A0%94/2024%E9%A1%B9%E7%9B%AE/textbookwriting/image/Draft_v1/1720875506415.png)

1. 在第三方库https://www.lfd.uci.edu/~gohlke/pythonlibs/中下载适合电脑的版本的whl文件，然后使用pip完成本地安装。
   win+R，将路径切换到上述whl文件所在路径，使用pip install ….whl完成安装；另一种方法是使用conda install rpy2 完成安装。
2. windows电脑配置环境变量。
   控制面板->系统和安全->系统->高级系统设置->环境变量，新建R的环境变量R_HOME，变量值为R所在的安装目录。
   再建一个rpy2的环境变量R_USER，变量值为rpy2的路径，如D:\Anaconda3\Lib\site-packages\rpy2
3. 接下来可以通过导入rpy2的包，看是否安装成功并可使用

```python
import rpy2.robjects as robjects
PI=robjects.r['pi']
print(PI[0])   ###打印出3.141592653589793
```

要注意：robjects.r("r_script") 可以执行r代码，比如 pi = robjects.r('pi') 就可以得到 R 中的PI（圆周率）， **返回的变量pi是一个向量，或者理解为python中的列表，通过pi[0] 就可以取出圆周率的值** 。

以下为使用python调用R函数并进行SFA分析输出效率的实例:

```python
import pandas as pd
import os
os.chdir(r'D:\yunpan\工作-创新科研\2024项目\ENG_SUPER\Program\R')
import rpy2.robjects as robjects
# creat an R function，自定义R函数  efunc为SFA的函数形式
robjects.r('''
library(frontier)
library( "plm" )
library(readxl)
library(writexl)
bank_data <- read_excel("D:/yunpan/工作-创新科研/2024项目/ENG_SUPER/program/R/bankdata_2024_6.xlsx",sheet = 'Sheet1')
# Error Components Frontier (Battese & Coelli 1992)
# with time-variant efficiencies
split(bank_data, bank_data$year) -> list_of_dfs
efunc <- function(x){banksfa <- sfa( log(netprofit)~ log(ie/save) +log(save)+ log(oec)+ log(ccloss)+ log(npl/loan) +log(loan), data = list_of_dfs[[x]])
         return(efficiencies(banksfa))}
           ''')
efflist = []
for i in range(2,19):
    t=robjects.r.efunc(i)
    efflist.append(t[:])
luckeff = pd.DataFrame(efflist)
luckeff.to_excel('efflist1.xlsx')
```

#### 14.5 DEA-Malmquist 模型

当被评价 DMU 的数据为包含多个时间点的观测值的面板数据时，就可以对生产率的变动情况以及变动的分解因素进行分析。DEA 模型的计算结果是技术效率，是一种相对效率，它不能用来动态地分析生产率的变化。这是因为它在每个时间点构造一个生产前沿，DMU 在不同时间点参考的生产前沿不一样。Malmquist 全要素生产率（Total Factor Productivity, TFP）指数就是常用来动态地分析生产率的变动情况，并对技术效率和技术进步各自对生产率变动所起的作用进行分析。

Malmquist 全要素生产率指数的概念最早源于 Malmquist（1953）。Färe 等人（1992）最早采用 DEA 的方法计算 Malmquist 指数，并将 Malmquist 指数（MI）分解为被评价 DMU 在两个时期内的技术效率变化（Technical Efficiency Change， EC）和生产技术的变化（TechnologicalChange，TC），其中生产技术的变化反映的是生产前沿的变动，技术效率的变化反映的是向生产前沿的移动程度。Chung 等（1997）将包含坏产出的方向距离函数应用于 Malmquist 模型，并将得出的 Malmquist 指数称为 Malmquist-Luenberger 指数。通常使用两个Malmquist-Luenberger 生产率指数的几何平均值得到以t期为基期到t +1期的全要素生产率的变化。

$ML_i^{t + 1} = {(ML_i^t \times ML_i^{t + 1})^{{1 \over 2}}} = MLTECH_i^{t + 1} \times MLEFFCH_i^{t + 1}$

Malmquist-Luenberger生产率被拆分为两部分:技术进步率和技术效率变化。Malmquist-Luenberger生产率指数值等于1表示，生产率没有发生变化;大于1或小于1时,分别表示生产率增长或生产率衰退。技术进步率指数测度环境生产前沿面从t到t+1时期的移动。如果指数值大于（小于）1,表示从t到t + 1时期环境生产前沿朝“更多（少）的好产出，更少（多）的坏产出”方向移动。出现了技术进步(技术退步)。技术效率变化指数测度从t到t + 1时期每个观察个体的实际生产与环境生产前沿面所示的最大可能产出迫近（Catching-up)程度的变化。如果指数值大于(小于) 1,表示从t到t + 1时期出现了效率提高(效率损失)。Malmquist-Luenberger生产率指数求解过程中需要借助线形规划计算，具体公式以及其他说明可以见相关文献。

##### 示例4：计算Malmquist指数

以下示例为计算多个DMU在多个时间点的TC,EC以及Malmquist指数

```python
import numpy as np
import random
import pandas as pd
import scipy.optimize as op
import os

def ML_dea(data,in_var,out_var,dmutime,dmu,method):
    ####基本信息
    yearinfo = data[dmutime].unique()
    dmuinfo = data[dmu].unique()  
    def splitbytime(data,dmutime):
        splitlistbytime=[]
        yearinfo = data[dmutime].unique()
        for i in yearinfo:
            rowx = data[(data[dmutime]==i)].index.tolist()
            datax = data.iloc[rowx,:].reset_index(drop=True)
            splitlistbytime.append(datax)
        return splitlistbytime

    def splitbydmu(data,dmu):
        splitlistbydmu=[]
        dmuinfo = data[dmu].unique()
        for i in dmuinfo:
            rowx = data[(data[dmutime]==i)].index.tolist(drop=True)
            datax = data.iloc[rowx,:].reset_index()
            splitlistbydmu.append(datax)
        return splitlistbydmu  

    datalist = splitbytime(data,dmutime)
    dmulist =  splitbytime(data,dmu)
    efflist11 = []
    for i in range(len(datalist)):
        s = ccr_dea(datalist[i], in_var=['I','N'], out_var=['O','U'], dmu = 'dmu', dmutime='year',method= method)
        efflist11.append(s)
    add_dmu_list=[] 
    for i in range(len(dmulist)):  
        if dmulist[i].shape[0]>1:  
            for j in range(dmulist[i].shape[0]):
                dmu_i = dmulist[i].loc[j].to_frame()
                add_dmu_list.append(dmu_i.T)

    eff12 = pd.DataFrame(columns = ['dmu','year','eff12'],index=range(len(add_dmu_list))) 
    eff21 = pd.DataFrame(columns = ['dmu','year','eff21'],index=range(len(add_dmu_list)))  
    effindex12 = 0
    effindex21 = 0   
    for i in range(len(add_dmu_list)): 
        dataadd =  add_dmu_list[i]
        comparetime = dataadd[dmutime].values  
        for jj in range(len(yearinfo)):
            if comparetime == yearinfo[jj]-1:  
                newdata = pd.concat([datalist[jj],dataadd]).reset_index(drop=True)
                midres =   ccr_dea(datalist[i], in_var=['I','N'], out_var=['O','U'], dmu = 'dmu', dmutime='year',method= method)
                eff12.loc[effindex12,'dmu']=midres.iloc[-1,0]
                eff12.loc[effindex12,'year']=midres.iloc[-1,1]
                eff12.loc[effindex12,'eff12']=midres.iloc[-1,2]
                effindex12 +=1
    for i in range(len(add_dmu_list)): 
        dataadd =  add_dmu_list[i]
        comparetime = dataadd[dmutime].values 
        for kk in range(len(yearinfo)):
            if comparetime == yearinfo[kk]+1:  
                newdata1 = pd.concat([datalist[kk],dataadd]).reset_index(drop=True)
                midres =   ccr_dea(datalist[i], in_var=['I','N'], out_var=['O','U'], dmu = 'dmu', dmutime='year',method= method)
                eff21.loc[effindex21,'dmu']=midres.iloc[-1,0]
                eff21.loc[effindex21,'year']=midres.iloc[-1,1]
                eff21.loc[effindex21,'eff21']=midres.iloc[-1,2] 
                effindex21 +=1  
    eff11 = pd.DataFrame(columns = ['dmu','year','eff'])
    for i in range(len(efflist11)):  
        eff11 = pd.concat([eff11,efflist11[i]]) 
    eff11 = eff11.drop(eff11.columns[3:], axis=1)   
    dfeff = pd.merge(eff11,eff12,how='left', on =['dmu','year']) 
    dfeff = pd.merge(dfeff,eff21,how='left', on =['dmu','year'])
    dfeff.sort_values(by = ['dmu','year'],ascending=[True,True],ignore_index=True,inplace=True)  
    milist = []
    for i in range(1,dfeff.shape[0]): 
        miunitlist= [0]*5
        if dfeff.loc[i,'dmu'] == dfeff.loc[i-1,'dmu'] and \
        pd.notna(dfeff.loc[i-1,'eff12']) and pd.notna(dfeff.loc[i,'eff21']):
            miunitlist[0] = dfeff.loc[i,'dmu']
            miunitlist[1] = dfeff.loc[i,'year']
            miunitlist[2] = dfeff.loc[i,'eff']/dfeff.loc[i-1,'eff']
            miunitlist[3] = ((dfeff.loc[i-1,'eff']* dfeff.loc[i,'eff21'])/(dfeff.loc[i-1,'eff12']* dfeff.loc[i,'eff']))**0.5
            miunitlist[4] = miunitlist[2] * miunitlist[3]
            milist.append(miunitlist) 
    miresults = pd.DataFrame(milist,columns=['dmu','year','catchup','frontiershift','mi'])
    return miresults

os.chdir('E:\\valdata\Data_input') 
data = pd.read_excel('test.xlsx')
in_var = ['I','N'] 
out_var= ['O','U']
dmutime = 'year'
dmu = 'dmu'
miresult = ML_dea(data,in_var,out_var,bad_var,dmutime,dmu,method="highs")   
  
```

#### 14.5 练习


:::

MIT Licensed | Copyright © 2024-present by [Yun Liao ](mailto:james@x.cool)
:::
