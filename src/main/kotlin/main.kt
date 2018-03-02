fun main(args: Array<String>) {
    var counter = 0
    val net = ImageNetEvolution(listOf(8, 12, 160, 40, 10), 0.005)
    while (true) {
        println("ПОРЯДОК: ${counter++}")
        MNIST.createBatch(40)
        for (i in 0 until 4) {
            println("-> $i")
            net.batch = (0..9).map { MNIST.batch[it * 4 + i] }
            var population = evolute0(net, 160)
            population = net.dropout(population, 0.01)
            evolute1(net, population)
        }
        for (i in 0 until 2) {
            println("-> $i")
            net.batch = (0..9).map { listOf(MNIST.batch[it * 4 + i], MNIST.batch[it * 4 + i + 1]) }.flatMap { it }
            var population = evolute0(net, 80)
            population = net.dropout(population, 0.01)
            evolute1(net, population)
        }
        net.batch = MNIST.batch
        for (i in 0 until 10) {
            var population = evolute0(net, 40)
            population = evolute1(net, population)
            NetworkIO().save(population.first().nw, "nets/nw.net")
        }
        break
    }
}

private fun evolute1(net: ImageNetEvolution, population: List<Individual>): List<Individual> {
    net.mutantStrategy = { _, _ -> 0.1 }
    net.mutantGenRate = 0.001
    val population1 = net.evolute(100, population)
    testNet(population1.first().nw)
    return population1
}

private fun evolute0(net: ImageNetEvolution, populationSize: Int): List<Individual> {
    net.mutantStrategy = { epoch, epochSize -> (1.0 - epoch * 1.0 / epochSize) }
    val population1 = net.evolute(50, populationSize)
    testNet(population1.first().nw)
    return population1
}

