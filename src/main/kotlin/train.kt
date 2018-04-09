fun main(args: Array<String>) {
    val testNumbers = (0..4).toList()
    train(testNumbers, emptyList(), 7)
}

private fun train(testNumbers: List<Int>, trainLayers: List<Int>, rateCount: Int) {
    val net = ImageNetEvolution(rateCount)
    net.trainLayers = trainLayers
    val addBatchSize = 500
    net.name = "nets/nw.net"
    net.mutantStrategy = { e, _ ->
        when {
            e < 100 -> ((100-e)/100.0)
            else -> 0.2
        }
    }
    net.batch = MNIST.buildBatch(addBatchSize).filter { it.index in testNumbers }
    var res2 = NetworkIO().load(net.name)?.let { testMedianNet(it, net.batch) } ?: 0.0
    var errorCounter = 0
    for (i in 1..10) {
        println("$i) ${net.batch.size}")
        net.evolute(1000, 80, 10)
        val res = testMedianNet(net.leader!!.nw, net.batch)
        if (res > res2) {
            NetworkIO().save(net.leader!!.nw, net.name)
            println("SAVE")
            res2 = res
            errorCounter = 0
        } else {
            println("NO SAVE")
            if (errorCounter == 2) return
            errorCounter++
            NetworkIO().save(net.leader!!.nw, "nets/_nw.net")
        }
        net.batch = net.batch.union(MNIST.buildBatch(addBatchSize).filter { it.index in testNumbers }).toList()
    }
}