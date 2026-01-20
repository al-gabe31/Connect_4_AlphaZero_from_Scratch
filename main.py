# this file is meant for testing code

from neural_network import *

activation_func = lambda x: 1 / (1 + pow(math.e, -1 * x))

# # first layer of nodes
# n11 = Node(0.35, 'n11')
# n12 = Node(0.7, 'n12')

# # second layer of nodes
# n21 = Node(alias='n21', activation_function=activation_func)
# n22 = Node(alias='n22', activation_function=activation_func)

# # n21.connect_preceding_nodes([n11], [0.2])
# # n21.connect_preceding_nodes([n12], [0.2])
# # n22.connect_preceding_nodes([n11, n12], [0.3, 0.3])

# n11.connect_suceeding_nodes([n21, n22], [0.2, 0.3])
# n12.connect_suceeding_nodes([n21, n22], [0.2, 0.3])

# # third layer of nodes
# n31 = Node(alias='n31', activation_function=activation_func)

# # n31.connect_preceding_nodes([n21, n22], [0.3, 0.9])

# n21.connect_suceeding_nodes([n31], [0.3])
# n22.connect_suceeding_nodes([n31], [0.9])



# print(n11)
# print(n12)
# print(n21)
# print(n22)
# print(n31)

# print(n11.preceding_conns)
# print(n11.suceeding_conns)

# print(n12.preceding_conns)
# print(n12.suceeding_conns)

# print(n21.preceding_conns)
# print(n21.suceeding_conns)

# print(n22.preceding_conns)
# print(n22.suceeding_conns)

# print(n31.preceding_conns)
# print(n31.suceeding_conns)



# 1st layer
n11 = Node(0, 'n11')
n12 = Node(1, 'n12')
n13 = Node(1, 'n13')

# 2nd layer
n21 = Node(alias='n21')
n22 = Node(alias='n22')
n23 = Node(alias='n23')

n21.connect_preceding_nodes([n11, n12, n13], [-2.2, -3.1, -4.6])
n22.connect_preceding_nodes([n11, n12, n13], [3.3, 2.7, 4.3])
n23.connect_preceding_nodes([n11, n12, n13], [-2.5, 4.0, 3.8])

# 3rd layer
n31 = Node(alias='n31')

n31.connect_preceding_nodes([n21, n22, n23], [0, 2.7, -1.6])



print(n11)
print(n12)
print(n13)
print(n21)
print(n22)
print(n23)
print(n31)





print('ran without error')