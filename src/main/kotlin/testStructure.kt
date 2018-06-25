import java.util.*

fun testMedianNet(number: Int, nw: Network, batch: Set<Image>, alpha: Double=15.0): Double {
    return batch.groupBy { it.index }.map { (i, list) ->
        val r = list.map { nw.activate(it, alpha)[0] }.average()
        if (i == number) r else 1 - r
    }.reduce { acc, d -> acc*d }
//    val r = b.map {
//        val o = nw.activate(it, 15.0).first()
//        if (it.index == number)  {
//            if (o > 0.5) 1 else 0
//        } else {
//            if (o <= 0.5) 1 else 0
//        }
//    }.sum()
//    val y = r.toDouble() / b.size
//    println("средний успех: ${(y*1000).toInt()/10.0}%")
//    return (y*10000).toInt() / 10000.0
}

fun main(args: Array<String>) {
    while (true) {
        print("Размер батча: ")
        val size = Scanner(System.`in`).nextInt()
        if (size < 30) return
        val name = "nw.net"
        CNetwork().load("nets/$name")?.let {
            val r = testMedianNet(number, it, MNIST.buildBatch(size))
            println(r)
        }
    }
}