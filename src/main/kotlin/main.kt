fun main(args: Array<String>) {
//    val net = ImageNetEvolution(listOf(8, 12, 160, 40, 10), 0.01)=3
//    val net = ImageNetEvolution(listOf(8, 12, 160, 60, 10), 0.01)+4
    val net = ImageNetEvolution(listOf(8, 14, 180, 60, 10), 0.01) //++5
//    val net = ImageNetEvolution(listOf(10, 14, 180, 60, 10), 0.01)+++6
    val populationSize = 80
    for (i in 1..10) {
        net.batch = MNIST.buildBatch(100)//.filter { it.index in listOf(0, 1, 2, 3, 4, 5, 6) }
        evolute(net, populationSize)
        val r = testMedianNet(net.leader!!.nw, net.batch)
        println("=>$r ($i)")
        //if (r > 0.7)
//            break
    }
}

private fun train(net: ImageNetEvolution, populationSize: Int): List<Individual> {
    var lastRate: Double
    var curRate = 1.0
    var population: List<Individual>
    do {
        lastRate = curRate
        population = evolute(net, populationSize)
        curRate = net.leader!!.rate
    } while (curRate < 0.99*lastRate)
    return population
}

private fun evolute(net: ImageNetEvolution, populationSize: Int): List<Individual> {
    var population = evolute0(net, populationSize)
    population = net.dropout(population, 0.01)
    evolute1(net, population, 100)
    NetworkIO().save(net.leader!!.nw, "nets/nw.net")
    return population
}

private fun evolute1(net: ImageNetEvolution, population: List<Individual>, epochSize: Int): List<Individual> {
    net.mutantStrategy = { _, _ -> 0.1 }
    net.mutantGenRate = 0.001
    val population1 = net.evolute(epochSize, population, 5)
    testNet(population1.first().nw, net.batch)
    return population1
}

private fun evolute0(net: ImageNetEvolution, populationSize: Int): List<Individual> {
    net.mutantStrategy = { epoch, epochSize -> (1.0 - epoch * 1.0 / epochSize) }
    val population1 = net.evolute(20, populationSize)
    testNet(population1.first().nw, net.batch)
    return population1
}

