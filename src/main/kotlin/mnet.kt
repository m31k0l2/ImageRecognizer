fun main(args: Array<String>) {
    val batch = MNIST.buildBatch(100)
    var names = listOf("02", "13", "24", "35", "46", "57", "68", "79", "80", "91")
    names = names.union(names.map { "${it}_1" }).toList()
    val nets = names.map { "nets/nw$it.net" }.map { NetworkIO().load(it)!! }
    var counter = 0
    batch.forEach {
        val results = nets.mapNotNull { nw ->
            val o = nw.activate(it)
            val x = o.max()!!
            if (x > 0.5) {
                o.indexOf(x)
            } else {
                null
            }
        }
        val r = (0..9).map { n -> n to results.filter { it == n }.size }.maxBy { it.second }?.first
        println("${it.index} -> $r")
        counter += if (it.index == r) 1 else 0
    }
    println("$counter / ${batch.size}")
//    val nw = NetworkIO().load("nets/nw.net", false)!! as MNetwork
//    batch.forEach {
//        val o = nw.activate(it)
//        val x = o.max()!!
//        print("${it.index} -> ")
//        if (x > 0.5) {
//            println(o.indexOf(x))
//        } else {
//            println("?")
//        }
//        println(o)
//    }
}