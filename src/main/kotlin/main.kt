fun main(args: Array<String>) {
    var counter = 0
    val net = ImageNetEvolution(listOf(8, 12, 160, 40, 10), 0.005)
    while (true) {
        println("ПОРЯДОК: ${counter++}")
        MNIST.createBatch(40)
        for (i in 0 until 4) {
            for (j in 1..5) {
                println("->$i->$j")
                net.batch = (0..9).map { MNIST.batch[it * 4 + i] }
                var population = evolute0(net, 40)
                println("->$i->$j")
                population = evolute1(net, population)
                println("->$i->$j")
                evolute2(net, population)
            }
        }
        net.batch = MNIST.batch
        var population = evolute0(net, 20)
        println("batch 40")
        population = evolute1(net, population)
        println("batch 40")
        population = evolute2(net, population)
        println("batch 40")
        NetworkIO().save(population.first().nw, "nets/nw.net")
        break
    }
}

private fun evolute1(net: ImageNetEvolution, population: List<Individual>): List<Individual> {
    net.mutantStrategy = {_, _ -> 1.0}
    net.mutantGenRate = 0.001
    val population1 = net.evolute(50, population, 10)
    testNet(population1.first().nw)
    return population1
}

private fun evolute2(net: ImageNetEvolution, population: List<Individual>): List<Individual> {
    net.mutantStrategy = { _, _ -> 0.01 }
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

