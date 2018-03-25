import java.util.*

class ImageNetEvolution: NetEvolution(mutantGenRate = .01) {
    override fun createNet() = CNetwork(1, 2, 10)
}

class ImageRecognizerEvolution: NetEvolution(mutantGenRate = .01) {
    override fun createNet() = MNetwork(10)
}

fun main(args: Array<String>) {
    while (true)
    trainFragments()
//    trainTotal()
}

private fun trainFragments() {
    for (i in 0..9) {
        (0..9).filter { it != i }.forEach {
            Trainer.train(ImageNetEvolution(), settings.populationSize, "nets/nw$i${it}_1.net", listOf(i, it))
        }
    }
}

fun trainTotal() {
    val net = ImageRecognizerEvolution()
    val populationSize = 80
    net.name = "nets/nw.net"
    net.batch = MNIST.buildBatch(30)
    net.leader = Individual(NetworkIO().load(net.name, false)!!)
    while(true) {
        Trainer.evolute(net, populationSize)
        val r = testMedianNet(net.leader!!.nw, net.batch)
        println("=>$r")
        NetworkIO().save(net.leader!!.nw, net.name)
        if (r > 0.95) break
    }
}

object Trainer {
    fun train(net: ImageNetEvolution, populationSize: Int, name: String, trainValues: List<Int>) {
        net.name = name
        for (i in 1..settings.trainCount) {
            println("train $name")
            net.batch = MNIST.buildBatch(settings.batchSize).filter { it.index in trainValues }
            (net.leader?.nw ?: NetworkIO().load(name))?.let {
                val r = testMedianNet(it, net.batch)
                if (r > 0.95) return
            }
            evolute(net, populationSize, 1000)
            NetworkIO().save(net.leader!!.nw, name)
        }
    }

    fun evolute(net: NetEvolution, populationSize: Int, epochSize: Int = settings.epochSize): List<Individual> {
        var population = evolute0(net, populationSize)
        population = net.dropout(population, 0.01)
        evolute1(net, population, epochSize)
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
}

