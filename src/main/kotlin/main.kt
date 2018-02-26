fun main(args: Array<String>) {
    var counter = 0
    while (true) {
        if (counter > 0) testNet("nets/nw.net")
        println("ПОРЯДОК: ${counter++}")
        val net = ImageNetEvolution(40, 10, listOf(8, 12, 160, 40, 10), 0.1)
        val nw = net.evolute(50, 400)
        testNet(nw)
        NetworkIO().save(nw, "nets/nw.net")
    }
}

