fun main(args: Array<String>) {
    var counter = 0
    while (true) {
//        MNIST.createBatch(40)
        if (counter > 0) testNet("nets/nw.net")
        println("ПОРЯДОК: ${counter++}")
        val net = ImageNetEvolution(120, 1, 0.01)
        val nw = net.evolute(200)
        testNet(nw)
        NetworkIO().save(nw, "nets/nw.net")
    }
}

