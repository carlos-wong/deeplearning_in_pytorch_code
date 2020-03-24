z = torch.zeros(2,5,3)
# print(z[0])

y = torch.ones(5,3)

x = torch.rand(5,3)
# print(x)
# print(x.t())
# # print(x)
# print(x[1,2])
# print(x[:,2])

z = x+y
# print(z)
# q = x.mm(y.t())
# print(q)


# a = np.ones(3)
# b = torch.from_numpy(a)
# np.add(a, 1, out=a)
# print(a)
# print(b)

# x_tensor = torch.rand(2,3)
# print(x_tensor)
# x_numpy = x_tensor.numpy()
# print(x_numpy)
# y_numpy = np.random.rand(2,3)
# print(y_numpy)
# y_tensor = torch.from_numpy(y_numpy)
# print(y_tensor)
# print(torch.cuda.is_available())


# x = Variable(torch.ones(2,2), requires_grad=True)
# print(x)

# y = x + 2
# print(y)

# z = y * y

# t = torch.mean(z)

# print(t)


s = Variable(torch.FloatTensor([[0.1,0.2]]),requires_grad=True)
x = Variable(torch.ones(2,2),requires_grad=True)
for i in range(10):
    s = s.mm(x)

z = torch.mean(s)

z.backward()

