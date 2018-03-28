object Detectors {
    private const val count = Settings.detectorChainCount
    private const val delta = 0.5
    private val detectors = (0 until count).map { k -> (0..9).map { n -> (0..9).map { "nets/nw$n${it}_$k.net" }.mapNotNull { NetworkIO().load(it) } } }
    private val backDetectors = (0 until count).map { k -> (0..9).map { n -> (0..9).map { "nets/nw$it${n}_$k.net" }.mapNotNull { NetworkIO().load(it) } } }
    private fun detectForward(k: Int, n: Int, image: Image) = detectors[k][n].map { it.activate(image) }.filter { it[n] > delta }.size
    private fun detectBack(k: Int, n: Int, image: Image) = backDetectors[k][n].map { it.activate(image) }.filter { it[n] > delta }.size

    private fun detect(k: Int, image: Image): Int? {
        val a = 7 // решение, если проголосовало больше a
        var r = (0..9).map { n -> Detectors.detectForward(k, n, image) }
        var m = r.max()!!
        if (m > a) {
            if (r.filter { it > a }.size == 1) return r.indexOf(m)
        }
        r = (0..9).map { n -> Detectors.detectBack(k, n, image) }
        m = r.max()!!
        if (m > a) {
            if (r.filter { it > a }.size == 1) return r.indexOf(m)
        }
        return null
    }

    private fun detect(image: Image): Int {
        val r = (0..2).map { k -> (0..9).map { n -> Detectors.detectForward(k, n, image) } }.reduce { acc, list ->
            acc.zip(list, { a, b -> a + b })
        }
        val m = r.max()!!
        return r.indexOf(m)
    }

    fun recognize(image: Image): Int {
        var d: Int? = null
        (0 until Detectors.count).takeWhile {
            d = Detectors.detect(it, image)
            d != null
        }
        return d ?: Detectors.detect(image)
    }
}