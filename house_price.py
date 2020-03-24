
x = torch.linspace(0, 100).type(torch.FloatTensor) #linspace可以生成0-100之间的均匀的100个数字\n
rand = torch.randn(100) * 10 #随机生成100个满足标准正态分布的随机数，均值为0，方差为1.将这个数字乘以10，标准方差变为10\n
y = x + rand #将x和rand相加，得到伪造的标签数据y。所以(x,y)应能近似地落在y=x这条直线上
