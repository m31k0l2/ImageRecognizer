import java.util.*

class ImageNetEvolution: NetEvolution(mutantGenRate = .01) {
    override fun createNet() = CNetwork(1, 2, 10)
}

class ImageRecognizerEvolution: NetEvolution(mutantGenRate = .01) {
    override fun createNet() = FNetwork(10)
}

fun main(args: Array<String>) {
//    while (true)
    trainFragments()
//    trainTotal()
}

fun trainTotal() {
    val net = ImageRecognizerEvolution()
    net.batch = MNIST.buildBatch(30)
    net.name = "nets/total.net"
    Trainer.evolute(net, 80)
    NetworkIO().save(net.leader!!.nw, net.name)
}

private fun trainFragments() {
    val trainSet = (102..987).mapNotNull {
        val a = it % 10
        val b = it/10 % 10
        val c = it/100 % 10
        listOf(a, b, c).sorted().takeIf { a != b && a != c && b != c }
    }.toSet()
    trainSet.forEach {
        Trainer.train(ImageNetEvolution(), 80, "nets/${it[0]}${it[1]}${it[2]}.net", it)
    }
}

object Trainer {
    fun train(net: ImageNetEvolution, populationSize: Int, name: String, trainValues: List<Int>) {
        net.name = name
        for (i in 1..Settings.trainCount) {
            println("train $name")
            net.batch = MNIST.buildBatch(Settings.batchSize).filter { it.index in trainValues }
            (net.leader?.nw ?: NetworkIO().load(name))?.let {
                val r = testMedianNet(it, net.batch)
                if (r > 0.95) return
            }
            evolute(net, populationSize, 1000)
            NetworkIO().save(net.leader!!.nw, name)
        }
    }

    fun evolute(net: NetEvolution, populationSize: Int, epochSize: Int = Settings.epochSize): List<Individual> {
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

