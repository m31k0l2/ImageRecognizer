import java.util.*

class ImageNetEvolution: NetEvolution(mutantGenRate = .01) {
    override fun createNet() = CNetwork(8, 14, 180, 60, 10)
    // 10, 14, 180, 60, 10 - 6
}

class ImageRecognizerEvolution: NetEvolution(mutantGenRate = .01) {
    override fun createNet() = MNetwork(60, 20, 10)
}

fun main(args: Array<String>) {
    val net = ImageNetEvolution()
    train(net, settings.populationSize, "nets/nw589.net", listOf(5, 8, 9))
}

private fun train(net: ImageNetEvolution, populationSize: Int, name: String, trainValues: List<Int>) {
    net.name = name
    for (i in 1..settings.trainCount) {
        println("train $name")
        net.batch = MNIST.buildBatch(settings.batchSize).filter { it.index in trainValues }
        (net.leader?.nw ?: NetworkIO().load(name))?.let {
            val r = testMedianNet(it, net.batch)
            if (r > 0.95) return
        }
        evolute(net, populationSize)
        NetworkIO().save(net.leader!!.nw, name)
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

private fun evolute(net: NetEvolution, populationSize: Int): List<Individual> {
    var population = evolute0(net, populationSize)
    population = net.dropout(population, 0.01)
    evolute1(net, population, settings.epochSize)
    return population
}

private fun evolute1(net: NetEvolution, population: List<Individual>, epochSize: Int): List<Individual> {
    net.mutantStrategy = { _, _ -> 0.2*Random().nextDouble() }
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

