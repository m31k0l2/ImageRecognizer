fun buildNetwork(structure: IntArray, alpha: Double)= network {
    convLayer(structure[0], CNetwork.cnnDividers[0], Pooler(CNetwork.poolerDividers[0]!!))
    convLayer(structure[1], CNetwork.cnnDividers[1], Pooler(CNetwork.poolerDividers[1]!!))
    convLayer(structure[2], CNetwork.cnnDividers[2])
    convLayer(structure[3], CNetwork.cnnDividers[3])
    fullConnectedLayer(structure[4]) {
        alpha(alpha)
    }
    fullConnectedLayer(structure[5]) {
        alpha(alpha)
    }
}

fun main(args: Array<String>) {
    val nw = buildNetwork(intArrayOf(6, 6, 4, 4, 40, 10), 1.0)
    println(nw.activate(MNIST.next()))
}