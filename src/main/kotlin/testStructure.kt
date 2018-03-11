fun testNet(nw: Network, batch: List<Image>): Double {
    var y = 0.0
    var counter = 0
    batch.sortedBy { it.index }.forEach {
        val o = nw.activate(it.colorsMatrix)
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
            val o = nw.activate(it.colorsMatrix)
            if (it.index >= o.size) .0
            else o[it.index]
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
    val batch = MNIST.buildBatch(1000)
    val nw = NetworkIO().load("nets/nw.net")!!
    testNet(nw, batch)
    testMedianNet(nw, batch)
//    val batchPairs = (0..9).map { batch.filter { i -> i.index == it } }
//    val n = 5
//    repeat(10, {
//        val rate = (0..9).map { batchPairs[it].shuffled() }.map { it.take(n).map { 1 - nw.activate(it.colorsMatrix)[it.index]  }.max()!! }.average()
//        println(1-rate)
//    })
}