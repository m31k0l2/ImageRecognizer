fun main(args: Array<String>) {
    var counter = 0
    while (true) {
        if (counter > 0) testNet("nets/nw.net")
        println("ПОРЯДОК: ${counter++}")
        MNIST.createBatch(10)
        var net = ImageNetEvolution(40, listOf(8, 12, 160, 40, 10), {epoch, epochSize -> (1.0 - epoch*1.0/epochSize)}, 0.1, 0.01)
        var population = net.evolute(50)
        testNet(population.first().nw)
        net = ImageNetEvolution(40, listOf(8, 12, 160, 40, 10), {_, _ -> 0.01}, 0.1, 0.01)
        population = net.evolute(50)
        testNet(population.first().nw)
        NetworkIO().save(population.first().nw, "nets/nw.net")
        break
    }
}

