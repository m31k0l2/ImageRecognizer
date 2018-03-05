import java.util.*

//fun main(args: Array<String>) {
//    val net = ImageNetEvolution(listOf(8, 12, 160, 40, 10), 0.005)
//    val batches = (1..4).map { MNIST.buildBatch(10) }
//    net.batch = batches[0]
//    train(net)
//    net.batch = batches[1]
//    train(net)
//    net.batch = listOf(batches[0], batches[1]).flatMap { it }
//    train(net)
//    net.batch = batches[1]
//    train(net)
//    net.batch = batches[2]
//    train(net)
//    net.batch = listOf(batches[1], batches[2]).flatMap { it }
//    train(net)
//    net.batch = batches[2]
//    train(net)
//    net.batch = batches[3]
//    train(net)
//    net.batch = listOf(batches[2], batches[3]).flatMap { it }
//    train(net)
//    net.batch = batches[0]
//    train(net)
//    net.batch = batches[1]
//    train(net)
//    net.batch = listOf(batches[0], batches[1]).flatMap { it }
//}

//fun main(args: Array<String>) {
//    val net = ImageNetEvolution(listOf(8, 12, 160, 40, 10), 0.005)
//    val batches = (1..4).map { MNIST.buildBatch(10) }
//    net.batch = batches[0]
//    val p1 = train(net, 80)
//    net.batch = batches[1]
//    val p2 = train(net, 80)
//    net.batch = batches[2]
//    val p3 = train(net, 80)
//    net.batch = batches[3]
//    val p4 = train(net, 80)
//    net.batch = listOf(batches[0], batches[1]).flatMap { it }
//    evolute1(net, p1.union(p2).toList())
//    val p5 = train(net, 40)
//    net.batch = listOf(batches[2], batches[3]).flatMap { it }
//    evolute1(net, p3.union(p4).toList())
//    val p6 = train(net, 40)
//    net.batch = batches.flatMap { it }
//    evolute1(net, p5.union(p6).toList())
//    train(net, 20)
//    NetworkIO().save(net.leader!!.nw, "nets/nw.net")
//}

fun main(args: Array<String>) {
    val net = ImageNetEvolution(listOf(8, 12, 160, 40, 10), 0.01)
    var counter = 0
    val batchSize = 20
    val populationSize = 60
    net.batch = MNIST.buildBatch(batchSize)
    train(net, populationSize)
    while (true) {
        counter++
        println("=> $counter")
        val item = Random().nextInt(10)
        println("now teach $item")
        net.batch = listOf(net.batch.filter { it.index != item }, MNIST.buildBatch(batchSize).filter { it.index == item }).flatMap { it }
        evolute(net, populationSize)
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
    population = evolute1(net, population)
    NetworkIO().save(net.leader!!.nw, "nets/nw.net")
    return population
}

private fun evolute1(net: ImageNetEvolution, population: List<Individual>): List<Individual> {
    net.mutantStrategy = { _, _ -> 0.1 }
    net.mutantGenRate = 0.001
    val population1 = net.evolute(100, population)
    testNet(population1.first().nw, net.batch)
    return population1
}

private fun evolute0(net: ImageNetEvolution, populationSize: Int): List<Individual> {
    net.mutantStrategy = { epoch, epochSize -> (1.0 - epoch * 1.0 / epochSize) }
    val population1 = net.evolute(50, populationSize)
    testNet(population1.first().nw, net.batch)
    return population1
}

