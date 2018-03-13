class ImageNetEvolution: NetEvolution(mutantGenRate = .01) {
    override fun createNet() = CNetwork(8, 14, 180, 60, 10)
    // 10, 14, 180, 60, 10 - 6
}

class ImageRecognizerEvolution: NetEvolution(mutantGenRate = .01) {
    override fun createNet() = MNetwork(60, 20, 10)
}

fun main(args: Array<String>) {
    val net = ImageNetEvolution()
    val populationSize = 80
    val prefix = "1"
    train(net, populationSize, "nets/nw02_$prefix.net", (0..2).toList())
    train(net, populationSize, "nets/nw13_$prefix.net", (1..3).toList())
    train(net, populationSize, "nets/nw24_$prefix.net", (2..4).toList())
    train(net, populationSize, "nets/nw35_$prefix.net", (3..5).toList())
    train(net, populationSize, "nets/nw46_$prefix.net", (4..6).toList())
    train(net, populationSize, "nets/nw57_$prefix.net", (5..7).toList())
    train(net, populationSize, "nets/nw68_$prefix.net", (6..8).toList())
    train(net, populationSize, "nets/nw79_$prefix.net", (7..9).toList())
    train(net, populationSize, "nets/nw80_$prefix.net", listOf(8, 9, 0))
    train(net, populationSize, "nets/nw91_$prefix.net", listOf(9, 0, 1))
}

private fun train(net: ImageNetEvolution, populationSize: Int, name: String, trainValues: List<Int>) {
    while (true) {
        net.batch = MNIST.buildBatch(100).filter { it.index in trainValues }
        evolute(net, populationSize, name)
        val r = testMedianNet(net.leader!!.nw, net.batch)
        println("=>$r")
        if (r > 0.95) break
    }
}

//fun main(args: Array<String>) {
//    val net = ImageRecognizerEvolution()
//    val populationSize = 100
//    while(true) {
//        net.batch = MNIST.buildBatch(200)
//        evolute(net, populationSize, "nets/nw.net")
//        val r = testMedianNet(net.leader!!.nw, net.batch)
//        println("=>$r")
//        if (r > 0.95) break
//    }
//}

private fun evolute(net: NetEvolution, populationSize: Int, name: String): List<Individual> {
    var population = evolute0(net, populationSize)
    population = net.dropout(population, 0.01)
    evolute1(net, population, 100)
    NetworkIO().save(net.leader!!.nw, name)
    return population
}

private fun evolute1(net: NetEvolution, population: List<Individual>, epochSize: Int): List<Individual> {
    net.mutantStrategy = { _, _ -> 0.1 }
    net.mutantGenRate = 0.001
    val population1 = net.evolute(epochSize, population, 5)
    testNet(population1.first().nw, net.batch)
    return population1
}

private fun evolute0(net: NetEvolution, populationSize: Int): List<Individual> {
    net.mutantStrategy = { epoch, epochSize -> (1.0 - epoch * 1.0 / epochSize) }
    val population1 = net.evolute(20, populationSize)
    testNet(population1.first().nw, net.batch)
    return population1
}

