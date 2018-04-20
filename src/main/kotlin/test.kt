import kotlin.system.measureNanoTime

fun main(args: Array<String>) {
//    val net = ImageNetEvolution(3)
//    net.trainLayers = (0..5).toList()
//    net.name = "nets/nw.net"
//    net.mutantStrategy = { _, _ -> .2 }
//    net.batch = MNIST.buildBatch(100)
//    val nw0 = NetworkIO().load(net.name)!!
//    val nw1 = net.mutate(Individual(nw0))!!.nw
//    val t1 = measureNanoTime {
//        for (i in 0..1000) {
//            val nw1 = net.cross(Individual(nw0) to Individual(nw1)).nw
//        }
//    }
//    val t2 = measureNanoTime {
//        for (i in 0..1000) {
//            val nw1 = net.cross2(Individual(nw0) to Individual(nw1)).nw
//        }
//    }
//    println((t1.toDouble() / t2.toDouble() - 1)*100)
}