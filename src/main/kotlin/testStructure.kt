import java.util.*

fun testNet(nw: Network, batch: List<Image>): Double {
    var y = 0.0
    var counter = 0
    batch.sortedBy { it.index }.forEach {
        val o = nw.activate(it)
        if (it.index >= o.size) return@forEach
        val result = o[it.index]
        if (result != o.max()) {
            println("${it.index} -> ${(result * 10000).toInt() / 100.0}% [${o.indexOf(o.max())} (${(o.max()!! * 10000).toInt() / 100.0}%)]")
        } else {
            counter++
            println("${it.index} -> ${(result * 10000).toInt() / 100.0}% [OK]")
        }
        y += result
    }
    y = (y*1000).toInt()/(batch.size*10.0)
    println("средний успех: $y%")
    println("$counter / ${batch.size}")
    return y
}

fun testMedianNet(nw: Network, batch: List<Image>): Double {
    val list = (0..9).map { i -> batch.filter { it.index == i } }.map {
        if (it.isEmpty()) return@map null
        it.map {
            nw.activate(it)[it.index]
        }.sorted()[it.size/2-1]
    }.filterNotNull()
    list.forEachIndexed { index, y ->
        val o = (y*1000).toInt()/10.0
        println("$index -> $o%")
    }
    val y = list.average()
    println("средний успех: ${(y*1000).toInt()/10.0}%")
    return y
}

fun main(args: Array<String>) {
    CNetwork.teachFromLayer = 6
    while (true) {
        print("nw.net? (y/n): ")
        val yesNo = Scanner(System.`in`).next()
        val name = "nw.net".takeIf { yesNo == "y" } ?: "_nw.net"
        print("Размер батча: ")
        val size = Scanner(System.`in`).nextInt()
        if (size < 30) return
        NetworkIO().load("nets/$name")?.let { testMedianNet(it, MNIST.buildBatch(size)) }
    }
//    testNet(NetworkIO().load("nets/nw01234.net")!!, MNIST.buildBatch(1000))
}